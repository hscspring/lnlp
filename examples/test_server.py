import os
import pytest
import requests

from hnlp import gen_input

MODEL_ROOT_TF = "./models/tf-models-tfserving/"


@pytest.fixture(scope="function")
def tfserving_payload(request):
    inp, y = gen_input(1, 60)
    inp_list = inp.numpy().tolist()
    if request.param == "inp":
        payload = {
            "inputs": {
                "input": inp_list,
                "training": False,
            },
            "signature_name": "serving_default",
        }
    elif request.param == "ins":
        payload = {
            "instances": inp_list,
            "signature_name": "serving_default",
        }
    else:
        raise NotImplementedError
    return payload


@pytest.fixture
def triton_payload():
    inp, y = gen_input(1, 60)
    inp_list = inp.numpy().tolist()
    payload = {
        "inputs": [{
            "name": "input",
            "shape": [1, 60],
            "datatype": "INT32",
            "data": inp_list
        }]
    }
    return payload


def get_curr_ver(model_root: str, model_name: str) -> str:
    model_dir = os.path.join(model_root, model_name)
    vers = os.listdir(model_dir)
    vers = [v.split(".")[0] for v in vers]
    ver = max(vers)
    return ver


@pytest.mark.parametrize("tfserving_payload,expected", [("ins", "predictions"),
                                                        ("inp", "outputs")],
                         indirect=["tfserving_payload"])
def test_tfserving_cls(tfserving_payload, expected):
    model_name = "mcls"

    # test basic api
    base_api = f"http://localhost:8501/v1/models/{model_name}"
    api1 = base_api + ":predict"
    resp1 = requests.post(api1, json=tfserving_payload)
    res1 = resp1.json()
    assert expected in res1

    # test api with version
    ver = get_curr_ver(MODEL_ROOT_TF, model_name)
    api2 = base_api + f"/versions/{ver}:predict"
    resp2 = requests.post(api2, json=tfserving_payload)
    res2 = resp2.json()
    assert expected in res2


def test_triton_cls(triton_payload):
    model_name = "mcls"

    # test basic api
    base_api = f"http://localhost:8000/v2/models/{model_name}"
    api1 = base_api + "/infer"
    resp1 = requests.post(api1, json=triton_payload)
    res1 = resp1.json()
    assert len(res1["outputs"][0]["data"]) == 15

    # test api with version
    ver = get_curr_ver(MODEL_ROOT_TF, model_name)
    api2 = base_api + f"/versions/{ver}/infer"
    resp2 = requests.post(api2, json=triton_payload)
    res2 = resp2.json()
    assert len(res2["outputs"][0]["data"]) == 15
