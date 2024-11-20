from django.apps import AppConfig
from .utils import create_pdf, create_index_if_not_exists

class TestAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'test_app'
    create_pdf()
    create_index_if_not_exists()