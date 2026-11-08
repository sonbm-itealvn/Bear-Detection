import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import torch.nn as nn

def test_relu_model_loads_coco_weights(tmp_path):
    from det.yolo.train_yolo import make_model
    y = make_model(tmp_path)
    m = y.model
    assert not [n for n, x in m.named_modules() if isinstance(x, nn.SiLU)]
    assert any(isinstance(x, nn.ReLU) for x in m.modules())
    assert m.model[-1].nc == 1
    # backbone conv weights came from yolov8n.pt (not the random init)
    import torch
    ref = torch.load("/data/nas07_new/PersonalData/phuocthien/yolov8n.pt", map_location="cpu", weights_only=False)["model"].float().state_dict()
    k = "model.0.conv.weight"
    assert torch.equal(m.state_dict()[k], ref[k])
