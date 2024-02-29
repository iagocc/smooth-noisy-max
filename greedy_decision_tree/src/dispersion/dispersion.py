class DispersionMeasure:
    def __call__(self, data: list):
        raise NotImplemented("The dispersion measure should be implemented.")
    
    def global_sensitivity(self, n: int):
        raise NotImplemented("The dispersion measure sensitivity should be implemented.")
    
    def local_sensitivity_at(self, n: int, t: int):
        raise NotImplemented("The dispersion measure sensitivity should be implemented.")

    def smooth_sensitivity(self, n: int, eps: int):
        raise NotImplemented("The dispersion measure sensitivity should be implemented.")
