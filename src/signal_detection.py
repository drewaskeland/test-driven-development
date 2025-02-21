import scipy.stats as stats  # Importing for norm.ppf (inverse normal CDF)

class SignalDetection:
    def __init__(self, hits, misses, fa, cr):
        self.hits = hits  # Hits: correctly detected signals
        self.misses = misses  # Misses: signals that were not detected
        self.fa = fa  # False alarms: noise mistaken for a signal
        self.cr = cr  # Correct rejections: correctly identified noise
#(ChatGPT assisted)
    def hit_rate(self):
        """Calculate the hit rate (H) = hits / (hits + misses)"""
        return self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

    def fa_rate(self):
        """Calculate the false alarm rate (FA) = fa / (fa + cr)"""
        return self.fa / (self.fa + self.cr) if (self.fa + self.cr) > 0 else 0

    def d_prime(self):
        """Calculate d-prime (d') = Z(Hit Rate) - Z(False Alarm Rate)"""
        H = self.hit_rate()
        FA = self.fa_rate()

#Avoid extreme probabilities (0 or 1) by applying a correction (ChatGPT assisted)
        H = min(max(H, 1e-5), 1 - 1e-5)
        FA = min(max(FA, 1e-5), 1 - 1e-5)

        return stats.norm.ppf(H) - stats.norm.ppf(FA)

    def criterion(self):
        """Calculate the criterion (C) = -0.5 * (Z(Hit Rate) + Z(False Alarm Rate))"""
        H = self.hit_rate()
        FA = self.fa_rate()

#Avoid extreme probabilities (0 or 1) by applying a correction
        H = min(max(H, 1e-5), 1 - 1e-5)
        FA = min(max(FA, 1e-5), 1 - 1e-5)

        return -0.5 * (stats.norm.ppf(H) + stats.norm.ppf(FA))

sd = SignalDetection(5, 2, 8, 2)
d_prime_value = sd.d_prime()
criterion_value = sd.criterion()
print(f"d': {d_prime_value}")
print(f"Criterion: {criterion_value}")