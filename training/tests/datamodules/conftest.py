import pytest

def pytest_addoption(parser):
    parser.addoption("--dataset", action="store", default="books3_1M_plus", help="Name of the dataset")
    parser.addoption("--chunk_name", action="store", default="chunk1", help="Name of the chunk")

@pytest.fixture
def dataset_name(request):
    return request.config.getoption("--dataset")

@pytest.fixture
def chunk_name(request):
    return request.config.getoption("--chunk_name")
