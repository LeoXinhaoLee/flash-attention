import pytest

def pytest_addoption(parser):
    parser.addoption("--dataset", action="store", default="books3_1M_plus", help="Name of the dataset")

@pytest.fixture
def dataset_name(request):
    return request.config.getoption("--dataset")
