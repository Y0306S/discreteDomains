"""CPU-only regression for FP32 LoRA loading into a BF16 base projection."""
import tempfile
import unittest


class AdapterPrecisionTest(unittest.TestCase):
    def test_text_checkpoint_keys_map_to_multimodal_base_losslessly(self):
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from safetensors.torch import load_file, save_file
        from pathlib import Path
        from mindware.domains.dlhpo.taskset_llm.llm_client import load_verified_adapter

        class TinyLayout(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layer = torch.nn.Module()
                layer.proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleList([layer])

        model = get_peft_model(TinyLayout(), LoraConfig(r=2, target_modules=["proj"]))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.fill_(.1234567)
        expected = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model).items()}
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            path = Path(tmp) / "adapter_model.safetensors"
            saved = load_file(str(path))
            save_file({key.replace("model.language_model.layers.", "model.layers."): value
                       for key, value in saved.items()}, str(path))
            loaded, report = load_verified_adapter(TinyLayout(), tmp)
        actual = get_peft_model_state_dict(loaded)
        self.assertTrue(report["key_mapping"])
        self.assertTrue(report["exact"])
        for key, value in expected.items():
            self.assertTrue(torch.equal(actual[key], value), key)

    def test_fp32_saved_weights_are_loaded_bit_for_bit(self):
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from mindware.domains.dlhpo.taskset_llm.llm_client import load_verified_adapter

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

        model = get_peft_model(Tiny(), LoraConfig(r=2, target_modules=["proj"]))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    # Deliberately choose values BF16 cannot represent exactly.
                    parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) * .000017 + .1234567)
        expected = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded, report = load_verified_adapter(Tiny(), tmp)
        actual = get_peft_model_state_dict(loaded)
        self.assertTrue(report["exact"])
        self.assertEqual(report["verified_tensors"], 2)
        for key, value in expected.items():
            self.assertEqual(actual[key].dtype, torch.float32)
            self.assertTrue(torch.equal(actual[key].detach().cpu(), value), key)
        self.assertEqual(loaded.base_model.model.proj.base_layer.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
