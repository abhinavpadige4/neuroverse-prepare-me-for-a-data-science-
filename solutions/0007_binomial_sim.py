import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simulate_binomial(n, p, num_simulations=10000, seed=42):
    """Simulate binomial distribution outcomes."""
    rng = np.random.default_rng(seed)
    results = rng.binomial(n, p, size=num_simulations)
    return results


def simulate_ab_test(n_users, p_control, p_treatment, num_simulations=10000, seed=42):
    """Simulate an A/B test using binomial distributions."""
    rng = np.random.default_rng(seed)
    control_conversions = rng.binomial(n_users, p_control, size=num_simulations)
    treatment_conversions = rng.binomial(n_users, p_treatment, size=num_simulations)
    return control_conversions, treatment_conversions


def compute_confidence_interval(conversions, n_users, confidence=0.95):
    """Compute confidence interval for conversion rate."""
    p_hat = conversions / n_users
    se = np.sqrt(p_hat * (1 - p_hat) / n_users)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    lower = p_hat - z * se
    upper = p_hat + z * se
    return lower, upper


def run_simulation():
    """Run complete binomial simulation for A/B testing."""
    n_users = 1000
    p_control = 0.10
    p_treatment = 0.12
    num_simulations = 10000

    print("=" * 60)
    print("BINOMIAL SIMULATION FOR A/B TESTING")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Users per group: {n_users}")
    print(f"  Control conversion rate: {p_control:.2%}")
    print(f"  Treatment conversion rate: {p_treatment:.2%}")
    print(f"  Number of simulations: {num_simulations}")

    control, treatment = simulate_ab_test(n_users, p_control, p_treatment, num_simulations)

    print(f"\n{'=' * 60}")
    print("SIMULATION RESULTS")
    print(f"{'=' * 60}")

    print(f"\nControl Group:")
    print(f"  Mean conversions: {control.mean():.2f}")
    print(f"  Std dev: {control.std():.2f}")
    print(f"  Mean conversion rate: {control.mean() / n_users:.4f}")

    print(f"\nTreatment Group:")
    print(f"  Mean conversions: {treatment.mean():.2f}")
    print(f"  Std dev: {treatment.std():.2f}")
    print(f"  Mean conversion rate: {treatment.mean() / n_users:.4f}")

    print(f"\n{'=' * 60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'=' * 60}")

    control_ci = compute_confidence_interval(control.mean(), n_users)
    treatment_ci = compute_confidence_interval(treatment.mean(), n_users)

    print(f"\nControl 95% CI: [{control_ci[0]:.4f}, {control_ci[1]:.4f}]")
    print(f"Treatment 95% CI: [{treatment_ci[0]:.4f}, {treatment_ci[1]:.4f}]")

    diff = treatment.mean() - control.mean()
    print(f"\nAbsolute lift: {diff:.2f} conversions")
    print(f"Relative lift: {diff / control.mean() * 100:.2f}%")

    print(f"\n{'=' * 60}")
    print("VISUALIZATION")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(control, bins=30, alpha=0.7, label='Control', color='blue')
    axes[0, 0].hist(treatment, bins=30, alpha=0.7, label='Treatment', color='red')
    axes[0, 0].set_title('Distribution of Conversions')
    axes[0, 0].set_xlabel('Number of Conversions')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

    axes[0, 1].hist(control / n_users, bins=30, alpha=0.7, label='Control', color='blue')
    axes[0, 1].hist(treatment / n_users, bins=30, alpha=0.7, label='Treatment', color='red')
    axes[0, 1].set_title('Distribution of Conversion Rates')
    axes[0, 1].set_xlabel('Conversion Rate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    lift = (treatment - control) / control
    axes[1, 0].hist(lift, bins=30, color='green', alpha=0.7)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Distribution of Relative Lift')
    axes[1, 0].set_xlabel('Relative Lift')
    axes[1, 0].set_ylabel('Frequency')

    positive_lift = (lift > 0).sum()
    axes[1, 1].pie([positive_lift, num_simulations - positive_lift],
                   labels=['Positive Lift', 'Negative/Zero Lift'],
                   autopct='%1.1f%%', colors=['green', 'red'])
    axes[1, 1].set_title('Proportion of Positive Lifts')

    plt.tight_layout()
    plt.savefig('binomial_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nPositive lift observed in {positive_lift / num_simulations * 100:.1f}% of simulations")
    print(f"\nSimulation complete. Plot saved to 'binomial_simulation.png'")


if __name__ == '__main__':
    run_simulation()
