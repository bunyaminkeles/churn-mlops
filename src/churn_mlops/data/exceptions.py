"""Veri katmanına ait özel hata sınıfları."""


class DataValidationError(Exception):
    """Verinin beklenen yapıya veya kurallara uymadığını belirtir."""
