Search.setIndex({"docnames": ["EnsembleXAI.Ensemble", "EnsembleXAI.Metrics", "Example_notebook", "Installation", "index"], "filenames": ["EnsembleXAI.Ensemble.rst", "EnsembleXAI.Metrics.rst", "Example_notebook.ipynb", "Installation.rst", "index.rst"], "titles": ["EnsembleXAI.Ensemble module", "EnsembleXAI.Metrics module", "Example Usage on ImageNet dataset", "Installation Guide", "Welcome to EnsembleXAI\u2019s documentation!"], "terms": {"autoweight": [0, 4], "input": [0, 1, 2], "tensorortupleoftensorsgener": 0, "metric_weight": 0, "list": [0, 1], "float": [0, 1, 2], "metric": [0, 4], "option": [0, 1, 3], "callabl": [0, 1], "none": [0, 1, 2], "precomputed_metr": 0, "union": [0, 1], "ani": [0, 1], "ndarrai": 0, "tensor": [0, 1, 2], "aggreg": [0, 2], "explan": [0, 1], "weight": [0, 1, 2], "qualiti": 0, "measur": [0, 1], "thi": [0, 1, 2], "function": [0, 1, 2], "an": [0, 1], "implement": [0, 1], "algorithm": [0, 1], "publish": 0, "1": [0, 1, 2], "It": 0, "us": [0, 1, 2], "ensemble_scor": [0, 1, 4], "calcul": [0, 1, 2], "each": [0, 1, 2], "One": 0, "should": [0, 1, 2], "pass": [0, 1], "paramet": [0, 1], "form": 0, "tupl": 0, "dimens": [0, 1], "correspond": [0, 1], "observ": 0, "singl": [0, 1], "default": [0, 1, 2], "assess": 0, "ignor": 0, "when": [0, 1], "i": [0, 1, 2, 3], "everi": 0, "valu": [0, 1, 2], "can": [0, 1, 2], "precomput": 0, "argument": [0, 1, 2], "need": [0, 1], "3": [0, 1], "dimension": [0, 1], "format": 0, "where": [0, 1], "support": [0, 1], "ar": [0, 1, 2, 3], "numpi": [0, 2], "torch": [0, 1, 2], "return": [0, 1, 2], "arithmet": 0, "mean": [0, 2], "type": [0, 1], "basic": [0, 2, 4], "simpl": 0, "like": 0, "averag": [0, 1, 2], "supervisedxai": [0, 2, 4], "kernel": [0, 2], "ridg": [0, 2], "regress": [0, 2], "suitabl": 0, "mask": [0, 1, 2], "avail": [0, 2], "note": [0, 1], "normal": 0, "standard": [0, 1], "deviat": 0, "befor": 0, "ensur": 0, "compar": [0, 1], "refer": [0, 1], "bobek": [0, 1], "": [0, 1, 2], "ba\u0142aga": [0, 1], "p": [0, 1, 2], "nalepa": [0, 1], "g": [0, 1], "j": [0, 1, 2], "2021": [0, 1], "toward": [0, 1], "model": [0, 1], "agnost": [0, 1], "In": [0, 1, 2], "paszynski": [0, 1], "m": [0, 1], "kranzlm\u00fcller": [0, 1], "d": [0, 1], "krzhizhanovskaya": [0, 1], "v": [0, 1], "dongarra": [0, 1], "sloot": [0, 1], "ed": [0, 1], "comput": [0, 1], "scienc": [0, 1], "icc": [0, 1], "lectur": [0, 1], "vol": [0, 1], "12745": [0, 1], "springer": [0, 1], "cham": [0, 1], "http": [0, 1, 2, 3], "doi": [0, 1], "org": [0, 1], "10": [0, 1, 2], "1007": [0, 1], "978": [0, 1], "030": [0, 1], "77970": [0, 1], "2_4": [0, 1], "exampl": [0, 1, 4], "import": [0, 1], "from": [0, 1, 2, 3], "we": [0, 2], "have": [0, 1], "4": [0, 1, 2], "15": [0, 2], "channel": [0, 1], "imag": [0, 1], "size": [0, 1, 2], "32": 0, "x": [0, 2], "randn": 0, "2": [0, 1, 2], "evalu": 0, "rand": 0, "ensembled_explan": 0, "0": [0, 1, 2], "8": [0, 1, 2], "aggregating_func": [0, 2], "str": [0, 1, 2], "simplest": 0, "wai": 0, "provid": 0, "custom": [0, 2], "combin": 0, "get": 0, "one": [0, 1, 2], "string": 0, "avg": [0, 2], "min": [0, 2], "max": [0, 2], "captum": [0, 2], "attr": [0, 2], "integratedgradi": [0, 2], "gradientshap": 0, "salienc": 0, "net": 0, "imageclassifi": 0, "ig": 0, "attribut": [0, 2], "target": [0, 2], "sal": 0, "stack": [0, 1, 2], "dim": [0, 2], "agg": 0, "n_fold": [0, 2], "int": [0, 1], "shuffl": 0, "fals": [0, 1, 2], "random_st": 0, "train": [0, 2], "supervis": 0, "machin": 0, "learn": [0, 2], "sklearn": 0, "kernel_ridg": 0, "kernelridg": 0, "krr": [0, 2], "output": [0, 2], "y": 0, "k": 0, "fold": [0, 2], "split": [0, 1, 2], "gener": [0, 2], "without": [0, 2], "inform": 0, "leakag": 0, "intern": 0, "model_select": 0, "kfold": 0, "make": [0, 3], "shape": [0, 1, 2], "same": [0, 1], "number": [0, 1], "greater": [0, 1], "than": [0, 1], "equal": [0, 1], "leav": 0, "out": 0, "done": [0, 1], "np": [0, 2], "sampl": [0, 2], "If": [0, 1], "uniform": 0, "auto": [0, 2], "invers": 0, "proport": 0, "area": [0, 1, 2], "also": [0, 2], "arrai": [0, 2], "promot": 0, "smaller": 0, "true": [0, 1, 2], "onli": [0, 1, 2], "which": [0, 1, 2], "l": [0, 1], "zou": [0, 1], "et": [0, 1], "al": [0, 1], "explain": [0, 1], "ai": [0, 1], "xai": [0, 1, 2], "sever": [0, 1], "commun": [0, 1], "acquir": [0, 1], "pneumonia": [0, 1], "covid": [0, 1], "19": [0, 1, 2], "respiratori": [0, 1], "infect": [0, 1], "ieee": [0, 1], "transact": [0, 1], "artifici": [0, 1], "intellig": [0, 1], "1109": [0, 1], "tai": [0, 1], "2022": [0, 1], "3153754": [0, 1], "randint": 0, "low": 0, "high": 0, "krr_explan": 0, "f1_score": [1, 2, 4], "threshold": [1, 2], "f1": [1, 2], "score": [1, 2], "recal": [1, 2], "precis": [1, 2], "harmon": [1, 2], "accordance_recal": [1, 2, 4], "accordance_precis": [1, 2, 4], "rang": [1, 2], "scenario": 1, "critic": [1, 2], "perfectli": 1, "match": 1, "propos": 1, "n": [1, 2], "width": 1, "height": 1, "repres": 1, "correl": 1, "presenc": [1, 2], "consid": [1, 2], "pair": 1, "how": 1, "much": 1, "ha": [1, 2], "cover": [1, 2], "ensembl": [1, 4], "cross_2d": 1, "plus_2d": 1, "repeat": [1, 2], "b": 1, "20000001788139343": 1, "data": [1, 2], "similar": 1, "classif": 1, "task": 1, "whole": 1, "over": [1, 2], "found": [1, 2], "intersection_mask": [1, 4], "intersect": [1, 2], "two": 1, "2000": 1, "confidence_impact_ratio": [1, 2, 4], "images_tensor": 1, "predictor": 1, "explanation_threshold": 1, "replace_valu": 1, "compare_to": [1, 2], "same_predict": [1, 2], "chang": [1, 2], "probabl": [1, 2], "after": [1, 2], "hide": 1, "taken": 1, "account": 1, "ones": 1, "correspod": 1, "class": [1, 2], "predict": [1, 2], "origin": [1, 2], "Then": [1, 3], "best": 1, "all": 1, "meanwhil": 1, "situat": 1, "wa": 1, "close": 1, "obscur": [1, 2], "new_predict": [1, 2], "maxim": [1, 2], "irrelev": 1, "therefor": [1, 2], "modifi": 1, "On": 1, "other": 1, "hand": 1, "For": [1, 2], "while": [1, 2], "differ": 1, "usag": [1, 3, 4], "alwai": 1, "The": 1, "stand": 1, "typic": 1, "case": 1, "possibli": 1, "wrap": 1, "nn": [1, 2], "softmax": [1, 2], "minim": [], "replac": 1, "decid": 1, "whether": 1, "maximum": 1, "new": 1, "index": 1, "probability_origin": 1, "probability_hidden_area": 1, "_impact_ratio_help": 1, "wrapper": 1, "decision_impact_ratio": [1, 2, 4], "5": [1, 2], "zero": 1, "ex_explan": 1, "booltensor": 1, "def": [1, 2], "input_tensor": 1, "item": [1, 2], "val": 1, "els": [1, 2], "6": [1, 2], "19999998807907104": 1, "6000000238418579": 1, "consist": [1, 2, 4], "photo": [1, 2], "do": 1, "diverg": 1, "ident": 1, "greatli": 1, "requir": [1, 2, 3], "depth": 1, "rgb": 1, "most": 1, "stabil": [1, 2, 4], "stabl": 1, "tensor_to_list_tensor": [1, 4], "first": [1, 2], "matrix_2_norm": [1, 4], "norm": 1, "matric": 1, "half": [1, 2], "18761281669139862": 1, "ones2": 1, "critica": 1, "metrics_scor": 1, "sum": 1, "time": 1, "13": [1, 2], "11": [1, 2], "tensor1": 1, "tensor2": 1, "threshold1": 1, "threshold2": 1, "absolute_valu": 1, "bool": 1, "logic": 1, "absolut": 1, "second": [1, 2], "boolean": 1, "intersection_over_union": [1, 2, 4], "divis": 1, "union_mask": [1, 4], "cross_2d_smal": 1, "plus_2d_smal": 1, "7": [1, 2], "intersection_area": 1, "union_area": 1, "3333333432674408": [1, 2], "matrix1": 1, "matrix2": 1, "sum_dim": 1, "By": 1, "work": [1, 2], "last": 1, "extend": 1, "remain": 1, "either": 1, "matrix": 1, "omit": 1, "both": [1, 2], "posit": [1, 2], "neg": [1, 2], "except": 1, "remov": 1, "specifi": 1, "onez_2d": 1, "zeroz_2d": 1, "onez_3d": 1, "zeroz_3d": 1, "4495": 1, "8990": 1, "onez_4d": 1, "zeroz_4d": 1, "4772": 1, "replace_mask": [1, 4], "replacement_index": 1, "spot": 1, "4d": 1, "3d": 1, "given": 1, "along": 1, "fit": 1, "image_4d": 1, "index_3d": 1, "replaced_imag": 1, "images_to_compar": 1, "epsilon": 1, "500": 1, "kwarg": 1, "between": 1, "As": 1, "creat": 1, "enough": 1, "some": 1, "here": [1, 2], "mai": 1, "take": 1, "signific": 1, "amount": 1, "process": 1, "power": 1, "memori": 1, "obtain": 1, "write": 1, "handl": 1, "might": 1, "additionali": 1, "choic": 1, "carefulli": 1, "test": 1, "distanc": 1, "manual": 1, "recommend": [1, 3], "9": [1, 2], "explain_dummi": 1, "100": 1, "13370312750339508": 1, "length": 1, "product": 1, "result": [1, 2], "reduc": 1, "factor": 1, "start": 1, "thu": 1, "cannot": 1, "larger": 1, "A": 1, "dim1": 1, "dim2": 1, "stacked_tensor": 1, "ensemblexai": [2, 3], "pretrain": 2, "resnet50": 2, "gc": 2, "o": 2, "json": 2, "pil": 2, "urllib": 2, "request": 2, "matplotlib": 2, "pyplot": 2, "plt": 2, "color": 2, "linearsegmentedcolormap": 2, "torchvis": 2, "transform": 2, "f": 2, "resiz": 2, "centercrop": 2, "resnet50_weight": 2, "occlus": 2, "noisetunnel": 2, "visual": 2, "viz": 2, "find": 2, "proper": [], "name": 2, "id": 2, "under": 2, "link": 2, "urlopen": 2, "s3": 2, "amazonaw": 2, "com": [2, 3], "deep": 2, "imagenet_class_index": 2, "url": 2, "imagenet_classes_dict": 2, "helper": 2, "hasten": 2, "images_list": 2, "image_path": 2, "_crop": 2, "224": 2, "forward": 2, "_resiz": 2, "232": 2, "image_nam": 2, "listdir": 2, "open": 2, "append": 2, "load_al": 2, "classid": 2, "all_img": 2, "images_dir": 2, "all_img_org": 2, "all_ten": 2, "to_tensor": 2, "img": 2, "all_msk": 2, "masks_dir": 2, "tens_img": 2, "tens_msk": 2, "set": 2, "up": 2, "correct": 2, "path": 2, "imagenets50": 2, "lusseg": 2, "github": [2, 3], "io": 2, "pixel": 2, "perfect": 2, "50": 2, "sake": 2, "easier": 2, "n01491361": 2, "tiger": 2, "shark": 2, "input_dir": 2, "join": 2, "getcwd": 2, "sep": 2, "semi": 2, "segment": 2, "notebook": [2, 3], "all_imag": 2, "all_images_origin": 2, "all_tensor": 2, "all_mask": 2, "tensor_imag": 2, "tensor_mask": 2, "zip": 2, "cat": 2, "displai": 2, "to_pil_imag": 2, "our": 2, "three": 2, "eval": 2, "resnet_transform": 2, "pipelin": 2, "lambda": 2, "proper_data": 2, "_": 2, "pred": 2, "probs2": 2, "single_pr": 2, "unsqueez": 2, "single_data": 2, "integrated_gradi": 2, "attributions_ig": 2, "n_step": 2, "200": 2, "collect": 2, "transformed_img": 2, "default_cmap": 2, "from_list": 2, "blue": 2, "ffffff": 2, "25": 2, "000000": 2, "256": 2, "_1": 2, "visualize_image_attr_multipl": 2, "transpos": 2, "squeez": 2, "heat_map": 2, "original_imag": 2, "cmap": 2, "show_colorbar": 2, "outlier_perc": 2, "noise_tunnel": 2, "attributions_ig_nt": 2, "nt_sampl": 2, "nt_type": 2, "smoothgrad_sq": 2, "_2": 2, "attributions_occ": 2, "stride": 2, "sliding_window_shap": 2, "baselin": 2, "_3": 2, "cpu": 2, "detach": 2, "attributions_occ2": 2, "20": 2, "_4": 2, "attributions_occ_all_25": 2, "attributions_occ_all_15": 2, "attributions_ig_nt_al": 2, "45": 2, "across": 2, "12": 2, "03428497165441513": 2, "decis": 2, "impact": 2, "ratio": 2, "occlusion25": 2, "33": 2, "third": 2, "confid": 2, "impli": 2, "increas": 2, "14": 2, "004259963985532522": 2, "004277209285646677": 2, "accord": 2, "16": 2, "3734": 2, "5004": 2, "2527": 2, "17": 2, "0423": 2, "1679": 2, "1560": 2, "18": 2, "1734609603881836": 2, "iou": 2, "occlusion15": 2, "0009072763496078551": 2, "sample_xai": 2, "40": 2, "21": 2, "all_stack": 2, "aggreg_all1": 2, "aggreg_all2": 2, "aggreg_all3": 2, "explanations_al": 2, "22": 2, "018643686547875404": 2, "033208806067705154": 2, "04660104587674141": 2, "23": 2, "015071512199938297": 2, "022996004670858383": 2, "0329229012131691": 2, "Of": 2, "cours": 2, "mani": 2, "more": 2, "properli": 2, "present": 2, "proof": 2, "concept": 2, "24": 2, "re": 2, "1440405398607254": 2, "26": 2, "6804": 2, "5995": 2, "7374": 2, "0942": 2, "2281": 2, "1852": 2, "27": 2, "22546647489070892": 2, "28": 2, "style": 2, "fast": 2, "plot_explan": 2, "columns_nam": 2, "classes_predict": 2, "method": 2, "nrow": 2, "ncol": 2, "len": 2, "fig": 2, "ax": 2, "subplot": 2, "figsiz": 2, "col": 2, "col_nam": 2, "titl": 2, "set_text": 2, "enumer": 2, "xaxi": 2, "set_ticks_posit": 2, "yaxi": 2, "set_yticklabel": 2, "set_xticklabel": 2, "imshow": 2, "vmin": 2, "vmax": 2, "255": 2, "set_ylabel": 2, "larg": 2, "expl": 2, "sign": 2, "amin": 2, "visualize_image_attr": 2, "plt_fig_axi": 2, "use_pyplot": 2, "show": 2, "29": 2, "predicted_nam": 2, "gradient": 2, "blended_heat_map": 2, "see": 2, "expect": 2, "non": 2, "nearli": 2, "gave": 2, "cross": 2, "valid": 2, "virtual": 3, "environ": 3, "strongli": 3, "encourag": 3, "clone": 3, "repositori": 3, "commandlin": 3, "activ": 3, "applic": 3, "pip": 3, "r": 3, "correct_path": 3, "xai_ensemblings_bs_m": 3, "txt": 3, "download": [2, 3], "current": 3, "unavail": 3, "due": 3, "being": 3, "privat": 3, "git": 3, "matts0000": 3, "egg": 3, "subdirectori": 3, "To": 3, "run": 3, "document": 3, "creation": 3, "nbsphinx": 3, "extens": 3, "convert": 3, "html": 3, "seper": 3, "pandoc": 3, "conda": 3, "sphinx": 3, "sphinx_rtd_them": 3, "clean": 3, "doc": 3, "project": 3, "directori": 3, "instal": 4, "guid": 4, "imagenet": 4, "dataset": 4, "modul": 4, "point": 1, "part": 1, "natur": 2, "download_class_imag": 2, "class_id": 2, "masks_path": 2, "full_path": 2, "kaggle_path": 2, "ilsvrc": 2, "cl": 2, "loc": 2, "file_nam": 2, "file_name_jpeg": 2, "jpeg": 2, "kaggl": 2, "competit": 2, "c": 2, "object": 2, "local": 2, "challeng": 2, "releas": 2, "a0fe9d82231f9bc4787ee76e304dfa51": 2, "api": 2, "func": 2, "n01491361_1000": 2, "crucial": 2}, "objects": {"": [[4, 0, 0, "-", "EnsembleXAI"]], "EnsembleXAI": [[0, 0, 0, "-", "Ensemble"], [1, 0, 0, "-", "Metrics"]], "EnsembleXAI.Ensemble": [[0, 1, 1, "", "autoweighted"], [0, 1, 1, "", "basic"], [0, 1, 1, "", "supervisedXAI"]], "EnsembleXAI.Metrics": [[1, 1, 1, "", "F1_score"], [1, 1, 1, "", "accordance_precision"], [1, 1, 1, "", "accordance_recall"], [1, 1, 1, "", "confidence_impact_ratio"], [1, 1, 1, "", "consistency"], [1, 1, 1, "", "decision_impact_ratio"], [1, 1, 1, "", "ensemble_score"], [1, 1, 1, "", "intersection_mask"], [1, 1, 1, "", "intersection_over_union"], [1, 1, 1, "", "matrix_2_norm"], [1, 1, 1, "", "replace_masks"], [1, 1, 1, "", "stability"], [1, 1, 1, "", "tensor_to_list_tensors"], [1, 1, 1, "", "union_mask"]]}, "objtypes": {"0": "py:module", "1": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"]}, "titleterms": {"ensemblexai": [0, 1, 4], "ensembl": [0, 2], "modul": [0, 1], "metric": [1, 2], "exampl": 2, "usag": 2, "imagenet": 2, "dataset": 2, "import": 2, "necessari": 2, "packag": [2, 3], "load": 2, "imag": 2, "plot": 2, "model": 2, "singl": 2, "explan": 2, "all": 2, "3": 2, "creat": 2, "instal": 3, "guid": 3, "user": 3, "develop": 3, "addit": 3, "welcom": 4, "": 4, "document": 4, "get": 4, "start": 4, "submodul": 4, "index": 4}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"Installation Guide": [[3, "installation-guide"]], "User Installation": [[3, "user-installation"]], "Package Development Additional Installation": [[3, "package-development-additional-installation"]], "Welcome to EnsembleXAI\u2019s documentation!": [[4, "welcome-to-ensemblexai-s-documentation"]], "Getting Started:": [[4, null]], "Submodules:": [[4, null]], "Index": [[4, "index"]], "EnsembleXAI.Metrics module": [[1, "module-EnsembleXAI.Metrics"]], "EnsembleXAI.Ensemble module": [[0, "module-EnsembleXAI.Ensemble"]], "Example Usage on ImageNet dataset": [[2, "Example-Usage-on-ImageNet-dataset"]], "Imports of the necessary packages": [[2, "Imports-of-the-necessary-packages"]], "Loading images and plotting the examples": [[2, "Loading-images-and-plotting-the-examples"]], "Model Loading": [[2, "Model-Loading"]], "Single Explanations": [[2, "Single-Explanations"]], "Explanations for all 3 images": [[2, "Explanations-for-all-3-images"]], "Metrics for the created explanations": [[2, "Metrics-for-the-created-explanations"]], "Ensembles": [[2, "Ensembles"]], "Plotting all explanations": [[2, "Plotting-all-explanations"]]}, "indexentries": {"ensemblexai.ensemble": [[0, "module-EnsembleXAI.Ensemble"]], "autoweighted() (in module ensemblexai.ensemble)": [[0, "EnsembleXAI.Ensemble.autoweighted"]], "basic() (in module ensemblexai.ensemble)": [[0, "EnsembleXAI.Ensemble.basic"]], "module": [[0, "module-EnsembleXAI.Ensemble"], [4, "module-EnsembleXAI"]], "supervisedxai() (in module ensemblexai.ensemble)": [[0, "EnsembleXAI.Ensemble.supervisedXAI"]], "ensemblexai": [[4, "module-EnsembleXAI"]]}})