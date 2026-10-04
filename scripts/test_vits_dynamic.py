import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
from vits_dynamic import repair_vits_masks, validate_vits_masks


def graph():
    nodes, initializers, inputs, outputs = [], [], [], []
    for name, dims, axis, prefix in [
        ("pred_semantic", [1,1,"tokens"], 2, "/vq_model/enc_p/Less"),
        ("text_seq", [1,"phones"], 1, "/vq_model/enc_p/Less_1"),
        ("ref_audio", [1,"samples"], 1, "/vq_model/Less")]:
        inputs.append(h.make_tensor_value_info(name,T.FLOAT,dims))
        for suffix, value in [("zero",np.array(0,dtype=np.int64)),("one",np.array(1,dtype=np.int64)),("axis",np.array(axis,dtype=np.int64)),("axes",np.array([0],dtype=np.int64)),("frozen",np.array([[12]],dtype=np.int64))]:
            initializers.append(nh.from_array(value,name+suffix))
        nodes.extend([h.make_node("Shape",[name],[name+'shape']),h.make_node("Gather",[name+'shape',name+'axis'],[name+'length']),h.make_node("Range",[name+'zero',name+'length',name+'one'],[name+'range']),h.make_node("Unsqueeze",[name+'range',name+'axes'],[name+'positions']),h.make_node("Less",[name+'positions',name+'frozen'],[name+'mask'],name=prefix)])
        outputs.append(h.make_tensor_value_info(name+'mask',T.BOOL,[1,None]))
    return h.make_model(h.make_graph(nodes,'masks',inputs,outputs,initializers),opset_imports=[h.make_opsetid('',20)],ir_version=9)


class DynamicMaskTests(unittest.TestCase):
    def test_repair_and_variable_lengths(self):
        m=graph()
        with self.assertRaisesRegex(ValueError,'frozen'): validate_vits_masks(m)
        self.assertEqual(len(repair_vits_masks(m)),3)
        self.assertEqual(repair_vits_masks(m),[])
        onnx.checker.check_model(m)
        session=ort.InferenceSession(m.SerializeToString(),providers=['CPUExecutionProvider'])
        for length in [5,12,428,940]:
            masks=session.run(None,{'pred_semantic':np.zeros((1,1,length),np.float32),'text_seq':np.zeros((1,length),np.float32),'ref_audio':np.zeros((1,length),np.float32)})
            for mask in masks:
                self.assertEqual(mask.shape,(1,length))
                self.assertTrue(mask.all())

    def test_reject_missing_mask(self):
        m=graph();del m.graph.node[-1]
        with self.assertRaisesRegex(ValueError,'expected 3'):repair_vits_masks(m)

    def test_reject_static_range(self):
        m=graph();m.graph.node[2].input[1]='pred_semanticone'
        with self.assertRaisesRegex(ValueError,'not dynamic'):repair_vits_masks(m)


if __name__=='__main__':unittest.main()
