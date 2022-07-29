import os
import argparse
import numpy as np

import tvm
from tvm import relay, autotvm, auto_scheduler
from tvm.contrib import ndk


target = 'llvm -model=snapdragon835 -mtriple=arm-linux-android -mattr=+neon'
target_host = 'llvm -mtriple=aarch64-linux-android-g++'

model_dir = "models"
log_dir = "logs"


class ModelImporter(object):
    def available_models(self):
        import inspect
        models = []
        for method in inspect.getmembers(type(self)):
            if "import_" in method[0]:
                models.append(method[0].split("import_")[1])
        return models

    def __call__(self, model, *args, **kwargs):
        import inspect

        for method in inspect.getmembers(type(self)):
            if "import_" + model == method[0]:
                return method[1](self, *args, **kwargs)
        raise ValueError("import_" + model + " not found.")


    def get_onnx_from_tf1(self, model_url, filename, input_names, output_names, shape_override = None):
        tf_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + f"/{model_dir}/{filename}.pb"
        )

        from tvm.contrib import download
        download.download(model_url, tf_model_file)
        # converted using command line:
        # python -m tf2onnx.convert --graphdef mace_resnet-v2-50.pb --output mace_resnet-v2-50.onnx --inputs input:0[1,224,224,3] --outputs resnet_v2_50/predictions/Reshape_1:0
        onnx_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + f"/{model_dir}/{filename}.onnx")
        if os.path.exists(onnx_model_file) == False:
            import tf2onnx
            import tensorflow as tf
            try:
                tf_compat_v1 = tf.compat.v1
            except ImportError:
                tf_compat_v1 = tf
            # Tensorflow utility functions
            import tvm.relay.testing.tf as tf_testing

            with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
                graph_def = tf_compat_v1.GraphDef()
                graph_def.ParseFromString(f.read())
                #graph = tf.import_graph_def(graph_def, name="")
                # Call the utility to import the graph definition into default graph.
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)

                model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
                    name=filename, input_names=input_names, output_names=output_names,
                    shape_override = shape_override,
                    output_path=onnx_model_file)

        return onnx_model_file


    def get_graphdef_from_tf1(self, model_url, filename):
        graph_def = None
        tf_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + f"/{model_dir}/{filename}.pb"
        )

        from tvm.contrib import download
        download.download(model_url, tf_model_file)
        # converted using command line:
        # python -m tf2onnx.convert --graphdef mace_resnet-v2-50.pb --output mace_resnet-v2-50.onnx --inputs input:0[1,224,224,3] --outputs resnet_v2_50/predictions/Reshape_1:0
        onnx_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + f"/../{model_dir}/{filename}.onnx")
        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        # Tensorflow utility functions
        import tvm.relay.testing.tf as tf_testing

        with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        return graph_def


    def import_mace_mobilenet_v1(self, dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb"
        filename = "mace_mobilenet-v1-1.0"
        input_names = ["input:0"]
        output_names = ["MobilenetV1/Predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 224, 224, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return mod, params


    def import_mace_resnet50_v2(self, dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/resnet-v2-50/resnet-v2-50.pb"
        filename = "mace_resnet-v2-50"
        input_names = ["input:0"]
        shape_override = {"input:0": [1, 224, 224, 3]}
        output_names = ["resnet_v2_50/predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names, shape_override)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 224, 224, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)
        
        mod = relay.quantize.prerequisite_optimize(mod, params)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return mod, params


    def import_mace_inception_v3(self, dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/inception-v3/inception-v3.pb"
        filename = "mace_inception-v3"
        input_names = ["input:0"]
        output_names = ["InceptionV3/Predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 299, 299, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        mod = relay.quantize.prerequisite_optimize(mod, params)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return mod, params


    def import_mace_yolov3(self, target="llvm", dtype="float32"):
        model_url = "http://cnbj1.fds.api.xiaomi.com/mace/miai-models/yolo-v3/yolo-v3.pb"
        filename = "mace_yolo-v3"
        graph_def = self.get_graphdef_from_tf1(model_url, filename)
        shape_dict = {"input_1": (1, 416, 416, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["conv2d_59/BiasAdd","conv2d_67/BiasAdd","conv2d_75/BiasAdd"])

        # # We assume our model's heavily-layout sensitive operators only consist of nn.conv2d
        # desired_layouts = {'nn.conv2d': ['NCHW', 'default']}

        # # Convert the layout to NCHW
        # # RemoveUnunsedFunctions is used to clean up the graph.
        # seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
        #                                 relay.transform.ConvertLayout(desired_layouts)])
        # with tvm.transform.PassContext(opt_level=3):
        #     mod = seq(mod)

        mod = relay.quantize.prerequisite_optimize(mod, params)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)
        return mod, params


def get_args():
    models = ModelImporter().available_models()
    parser = argparse.ArgumentParser(description="Script for cross-compilation and collection inference statistics")
    subparsers = parser.add_subparsers(dest="parser", help='Script modes')

    tune_parser = subparsers.add_parser("tune", help="Tuning and compiling the base model for a device connected via RPC")

    tune_subparsers = tune_parser.add_subparsers(dest="tune_parser", help='Which auto-tuner to use')
    atvm_tune_parser = tune_subparsers.add_parser("atvm", help="Use AutoTVM")
    ansor_tune_parser = tune_subparsers.add_parser("ansor", help="Use Auto-scheduler")

    for _tune_parser in [atvm_tune_parser, ansor_tune_parser]:
        _tune_parser.add_argument("-m", "--model", type=str, choices=models, help="Model to tune")
        _tune_parser.add_argument("-t", "--type", type=str, default="float32", choices=["float32", "float16"],
            help="Specify whether the model should be run with single or half precision floating point values")

        _tune_parser.add_argument("-r", "--rpc_tracker_host", default=os.environ["TVM_TRACKER_HOST"], help="RPC tracker host IP address")
        _tune_parser.add_argument("-p", "--rpc_tracker_port", default=os.environ["TVM_TRACKER_PORT"], help="RPC tracker host port")
        _tune_parser.add_argument("-k", "--rpc_key", default="android", help="RPC key to use")

        _tune_parser.add_argument("-l", "--log", type=str, default=None, help="Path to an auto-tuning log file by AutoTVM")
        _tune_parser.add_argument("--disable_inference", action="store_true", 
            help="Disable collection of inference statistics on completion of tuning")

        _tune_parser.add_argument('-T', '--target', type=str, default=target, help='Compilation target')
        _tune_parser.add_argument('-H', '--target_host', type=str, default=target_host, help='Compilation host target')

    exec_parser = subparsers.add_parser("exec", help="Execute compiled model")
    exec_parser.add_argument('-m', '--model', required=True, type=str, help="path to compiled .so file")

    exec_parser.add_argument("-r", "--rpc_tracker_host", default=os.environ["TVM_TRACKER_HOST"], help="RPC tracker host IP address")
    exec_parser.add_argument("-p", "--rpc_tracker_port", default=os.environ["TVM_TRACKER_PORT"], help="RPC tracker host port")
    exec_parser.add_argument("-k", "--rpc_key", default="android", help="RPC key to use")

    args = parser.parse_args()

    if args.parser == "tune":
        args.name = '.'.join([args.rpc_key, args.model, args.type])
        if args.log is None:
            args.log = os.path.join(log_dir, f"/{args.name}.{args.tune_parser}.log")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

    if args.rpc_tracker_port is not None:
        args.rpc_tracker_port = int(args.rpc_tracker_port)       
    
    return args


args = get_args()

   
def downcast_fp16(func, module):
    from tvm.relay.expr_functor import ExprMutator
    from tvm.relay.expr import Call, Var, Constant, TupleGetItem
    from tvm.relay import transform as _transform
    from tvm.relay import cast
    from tvm.ir import IRModule
    from tvm.relay import function as _function

    """Downcast to fp16 mutator
    Parameters
    ---------
    graph: Function
        The original graph.
    Retruns
    -------
    The graph after dowmcasting to half-precision floating-point.
    """
    filter_list = ["vision.get_valid_counts", "vision.non_max_suppression"]

    class DowncastMutator(ExprMutator):
        """Downcast to fp16 mutator"""

        def visit_call(self, call):
            dtype = "float32" if call.op.name in filter_list else "float16"
            new_fn = self.visit(call.op)
            # Collect the original dtypes
            type_list = []
            if call.op.name in filter_list:
                # For NMS
                for arg in call.args:
                    if isinstance(arg, TupleGetItem) and isinstance(
                        arg.tuple_value, Call
                    ):
                        tuple_types = arg.tuple_value.checked_type.fields
                        type_list.append(tuple_types[arg.index].dtype)
                if call.op.name == "vision.get_valid_counts":
                    tuple_types = call.checked_type.fields
                    for cur_type in tuple_types:
                        type_list.append(cur_type.dtype)

            args = [self.visit(arg) for arg in call.args]
            new_args = list()
            arg_idx = 0
            for arg in args:
                if isinstance(arg, (Var, Constant)):
                    new_args.append(cast(arg, dtype=dtype))
                else:
                    if call.op.name in filter_list:
                        if (
                            isinstance(arg, TupleGetItem)
                            and type_list[arg_idx] == "int32"
                        ):
                            new_args.append(arg)
                        else:
                            new_args.append(cast(arg, dtype=dtype))
                    else:
                        new_args.append(arg)
                arg_idx += 1
            if (
                call.op.name in filter_list
                and call.op.name != "vision.get_valid_counts"
            ):
                return cast(Call(new_fn, new_args, call.attrs), dtype="float16")
            return Call(new_fn, new_args, call.attrs)

    class UpcastMutator(ExprMutator):
        """upcast output back to fp32 mutator"""

        def visit_call(self, call):
            return cast(call, dtype="float32")

    def infer_type(node, mod=None):
        """A method to infer the type of an intermediate node in the relay graph."""
        if isinstance(mod, IRModule):
            mod["main"] = _function.Function(relay.analysis.free_vars(node), node)
            mod = _transform.InferType()(mod)
            entry = mod["main"]
            ret = entry.body
        else:
            new_mod = IRModule.from_expr(node)
            if mod is not None:
                new_mod.update(mod)
                new_mod = _transform.InferType()(new_mod)
                entry = new_mod["main"]
                ret = entry if isinstance(node, _function.Function) else entry.body

        return ret

    func = infer_type(func, module)
    downcast_pass = DowncastMutator()
    func = downcast_pass.visit(func)
    upcast_pass = UpcastMutator()
    func = upcast_pass.visit(func)
    func = infer_type(func, module)
    new_mod = IRModule.from_expr(func)
    # new_mod.update(module)
    return new_mod


class Executor:
    def __init__(self):
        self.remote = None
        self.tracker = None
        if args.parser == "tune":
            self.target = args.target
            self.target_host = args.target_host

    def _connect_tracker(self):
        from tvm import rpc

        print(
            "Tracker attempting connection on {}:{}".format(
                args.rpc_tracker_host, args.rpc_tracker_port
            )
        )
        self.tracker = rpc.connect_tracker(args.rpc_tracker_host, args.rpc_tracker_port)
        self.remote = self.tracker.request(
            args.rpc_key, priority=0, session_timeout=None
        )
        print("Tracker connected to remote RPC server")

    def _disconnect_tracker(self):
        self.remote = None
        self.tracker = None

    def tune_kernels_autotvm(
        tasks, measure_option, tuner="gridsearch", n_trial=16, early_stopping=None, log_filename="tuning.log"
    ):
        from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
        
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            # create tuner
            if tuner == "xgb" or tuner == "xgb-rank":
                tuner_obj = XGBTuner(task, loss_type="rank")
            elif tuner == "ga":
                tuner_obj = GATuner(task, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(task)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(task)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            # do tuning
            n_trial = min(n_trial, len(task.config_space))
            print ("Number of trials as len of config_space: " + str(n_trial))
            
            tuner_obj.tune(
                n_trial=n_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename),
                ],
            )

    def tune_autotvm(self, mod, params):
        # extract workloads from relay program
        print("Extract autotvm tasks...")
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=self.target, target_host=self.target_host, params=params)
        for idx, task in enumerate(tasks):
            print("========== Task %d ==========" %
                    (idx))
            print(task)

        # run tuning tasks
        atvmMeasureOptions = autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
            runner=autotvm.RPCRunner(
                args.rpc_key,
                host=args.rpc_tracker_host,
                port=args.rpc_tracker_port,
                number=50,
                timeout=15,
                ),
            )

        log_file = os.path.join(log_dir, f"{args.name}.atvm.json")
        Executor.tune_kernels_autotvm(tasks,
            log_filename = log_file,
            tuner = "gridsearch",
            n_trial = 66,
            early_stopping = None,
            measure_option = atvmMeasureOptions)

    def compile_autotvm(self, mod, params):
        log_file = os.path.join(log_dir, f"{args.name}.atvm.json")
        with autotvm.apply_history_best(log_file):
            print("Compile...")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=self.target, target_host=self.target_host, params=params)
        lib.export_library(f"{args.name}.atvm.so", ndk.create_shared)

    def run_tuning(tasks, task_weights, log_file):
        print("Begin tuning...")
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        builder=auto_scheduler.LocalBuilder(build_func=ndk.create_shared, timeout=15)
        tune_option = auto_scheduler.TuningOptions(
            builder=builder,
            num_measure_trials=512,
            num_measures_per_round = 64, # to speed-up round-robin measurements
            runner=auto_scheduler.RPCRunner(
                args.rpc_key,
                host=args.rpc_tracker_host,
                port=args.rpc_tracker_port,
            ),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            # verbose=2
        )
        tuner.tune(tune_option)

    def tune_ansor(self, mod, params):
        log_file = os.path.join(log_dir, f"{args.name}.ansor.json")

        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], params, target=self.target, target_host=self.target_host)
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" %
                (idx, task.workload_key))
            print(task.compute_dag)

        Executor.run_tuning(tasks, task_weights, log_file)

    def compile_ansor(self, mod, params):
        log_file = os.path.join(log_dir, f"{args.name}.ansor.json")

        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target,
                                    target_host=target_host, params=params)

        lib.export_library(f"{args.name}.ansor.so", ndk.create_shared)
    
    def benchmark(self, input_path):
        from tvm.contrib import  graph_executor

        if self.remote == None:
            self._connect_tracker()

        print("Uploading binary...")
        self.remote.upload(input_path)
        lib = self.remote.load_module(os.path.basename(input_path))
        ctx = self.remote.cpu(0)
        m = graph_executor.GraphModule(lib["default"](ctx))

        print("Starting measurements...")
        ftimer = m.module.time_evaluator("run", ctx, repeat=10, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
        )

        self._disconnect_tracker()


def main():
    executor = Executor()
    if args.parser == "tune":
        importer = ModelImporter()
        mod, params = importer(args.model, dtype=args.type)
        
        if args.tune_parser == "atvm":
            executor.tune_autotvm(mod, params)
            executor.compile_autotvm(mod, params)
        elif args.tune_parser == "ansor":
            executor.tune_ansor(mod, params)
            executor.compile_ansor(mod, params)

        if args.disable_inference == False:
            executor.benchmark(input_path=f"{args.name}.{args.tune_parser}.so")
    
    elif args.parser == "exec":
        executor.benchmark(input_path=args.model)


if __name__ == "__main__":
    main()
