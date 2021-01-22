"""
TODO: description
"""

import os
import torch
import shutil
import numpy as np
from collections import OrderedDict
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d

do_check = False
try:
	import onnxruntime
	do_check = True
except ImportError:
	pass


def rename_nodes_onnx(onnx_model, current_inp_name, current_out_name, new_inp_name, new_out_name, quiet=True):
	for i in range(len(onnx_model.graph.node)):
		for j in range(len(onnx_model.graph.node[i].input)):
			if onnx_model.graph.node[i].input[j] in current_inp_name:
				print('-' * 60)
				print(onnx_model.graph.node[i].name)
				print(onnx_model.graph.node[i].input)
				print(onnx_model.graph.node[i].output)

				onnx_model.graph.node[i].input[j] = onnx_model.graph.node[i].input[j].split(':')[0]

		for j in range(len(onnx_model.graph.node[i].output)):
			if onnx_model.graph.node[i].output[j] in current_out_name:
				print('-' * 60)
				print(onnx_model.graph.node[i].name)
				print(onnx_model.graph.node[i].input)
				print(onnx_model.graph.node[i].output)

				onnx_model.graph.node[i].output[j] = onnx_model.graph.node[i].output[j].split(':')[0]

	for i in range(len(onnx_model.graph.input)):
		if onnx_model.graph.input[i].name in current_out_name:
			print('-' * 60)
			print(onnx_model.graph.input[i])
			onnx_model.graph.input[i].name = onnx_model.graph.input[i].name.split(':')[0]

	for i in range(len(onnx_model.graph.output)):
		if onnx_model.graph.output[i].name in current_inp_name:
			print('-' * 60)
			print(onnx_model.graph.output[i])
			onnx_model.graph.output[i].name = onnx_model.graph.output[i].name.split(':')[0]

	return onnx_model

if do_check:
	# требует onnxruntime
	def check_onnx_model(onnx_file, example, torch_out):
		ort_session = onnxruntime.InferenceSession(onnx_file)

		def to_numpy(tensor):
			return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

		# compute ONNX Runtime output prediction
		if isinstance(example, (tuple, list)):
			ort_inputs = {ort_session.get_inputs()[i_e].name: to_numpy(ex).astype(np.float32) for i_e, ex in enumerate(example)}
		else:
			ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(example).astype(np.float32)}
		ort_outs = ort_session.run(None, ort_inputs)

		# compare ONNX Runtime and PyTorch results
		np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

		print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def save_checkpoint(state, epoch, save_dir, is_best=False, remove_module_from_keys=False):
	"""
	"""
	os.makedirs(save_dir, exist_ok=True)
	if remove_module_from_keys:
		# remove 'module.' in state_dict's keys
		state_dict = state['state_dict']
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			if k.startswith('module.'):
				k = k[7:]
			new_state_dict[k] = v
		state['state_dict'] = new_state_dict
	# save
	fpath = os.path.join(save_dir, 'model.pth.tar-' + str(epoch))
	torch.save(state, fpath)
	print('Checkpoint saved to "{}"'.format(fpath))
	if is_best:
		shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'model-best.pth.tar'))