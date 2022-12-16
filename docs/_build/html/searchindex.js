Search.setIndex({"docnames": ["EnsembleXAI", "EnsembleXAI.Ensemble", "EnsembleXAI.Metrics", "Example_notebook", "Installation", "index"], "filenames": ["EnsembleXAI.rst", "EnsembleXAI.Ensemble.rst", "EnsembleXAI.Metrics.rst", "Example_notebook.ipynb", "Installation.rst", "index.rst"], "titles": ["EnsembleXAI package", "EnsembleXAI.Ensemble module", "EnsembleXAI.Metrics module", "Example Usage on ImageNet dataset", "User Installation", "Welcome to EnsembleXAI\u2019s documentation!"], "terms": {"ensembl": [0, 2, 5], "autoweight": [0, 1, 5], "basic": [0, 1, 3, 5], "supervisedxai": [0, 1, 3, 5], "metric": [0, 1, 5], "f1_score": [0, 2, 3, 5], "accordance_precis": [0, 2, 3, 5], "accordance_recal": [0, 2, 3, 5], "confidence_impact_ratio": [0, 2, 3, 5], "consist": [0, 2, 3, 5], "decision_impact_ratio": [0, 2, 3, 5], "ensemble_scor": [0, 1, 2, 5], "intersection_mask": [0, 2, 5], "intersection_over_union": [0, 2, 3, 5], "matrix_2_norm": [0, 2, 5], "replace_mask": [0, 2, 5], "stabil": [0, 2, 3, 5], "tensor_to_list_tensor": [0, 2, 5], "union_mask": [0, 2, 5], "input": [1, 2, 3], "tensorortupleoftensorsgener": 1, "metric_weight": 1, "list": [1, 2], "float": [1, 2, 3], "option": [1, 2, 4], "callabl": [1, 2], "none": [1, 2, 3], "precomputed_metr": 1, "union": [1, 2], "ani": [1, 2], "ndarrai": 1, "tensor": [1, 2, 3], "aggreg": [1, 3], "explan": [1, 2], "weight": [1, 2, 3], "qualiti": 1, "measur": [1, 2], "thi": [1, 2, 3], "function": [1, 2, 3], "an": [1, 2], "implement": [1, 2], "algorithm": [1, 2], "publish": 1, "1": [1, 2, 3], "It": 1, "us": [1, 2, 3], "calcul": [1, 2, 3], "each": [1, 2, 3], "One": 1, "should": [1, 2], "pass": [1, 2], "paramet": [1, 2], "form": 1, "tupl": 1, "dimens": [1, 2], "correspond": [1, 2], "observ": [1, 3], "singl": [1, 2], "default": [1, 2, 3], "assess": 1, "ignor": 1, "when": [1, 2], "i": [1, 2, 3, 4], "everi": 1, "valu": [1, 2, 3], "can": [1, 2, 3], "precomput": 1, "argument": [1, 2, 3], "need": [1, 2], "3": [1, 2], "dimension": [1, 2], "format": 1, "where": [1, 2], "support": [1, 2], "ar": [1, 2, 3, 4], "numpi": [1, 3], "torch": [1, 2, 3], "return": [1, 2, 3], "arithmet": 1, "mean": [1, 3], "type": [1, 2], "simpl": 1, "like": 1, "averag": [1, 2, 3], "kernel": [1, 3], "ridg": 1, "regress": 1, "suitabl": 1, "mask": [1, 2, 3], "avail": 1, "note": [1, 2], "normal": 1, "standard": [1, 2], "deviat": 1, "befor": 1, "ensur": 1, "compar": [1, 2], "refer": [1, 2], "bobek": [1, 2], "": [1, 2], "ba\u0142aga": [1, 2], "p": [1, 2], "nalepa": [1, 2], "g": [1, 2], "j": [1, 2, 3], "2021": [1, 2], "toward": [1, 2], "model": [1, 2], "agnost": [1, 2], "In": [1, 2, 3], "paszynski": [1, 2], "m": [1, 2], "kranzlm\u00fcller": [1, 2], "d": [1, 2, 3], "krzhizhanovskaya": [1, 2], "v": [1, 2], "dongarra": [1, 2], "sloot": [1, 2], "ed": [1, 2], "comput": [1, 2], "scienc": [1, 2], "icc": [1, 2], "lectur": [1, 2], "vol": [1, 2], "12745": [1, 2], "springer": [1, 2], "cham": [1, 2], "http": [1, 2, 3, 4], "doi": [1, 2], "org": [1, 2], "10": [1, 2, 3], "1007": [1, 2], "978": [1, 2], "030": [1, 2], "77970": [1, 2], "2_4": [1, 2], "exampl": [1, 2, 5], "import": [1, 2], "from": [1, 2, 3, 4], "we": [1, 3], "have": [1, 2], "4": [1, 2, 3], "15": [1, 3], "channel": [1, 2], "imag": [1, 2], "size": [1, 2, 3], "32": [1, 3], "x": [1, 3], "randn": 1, "2": [1, 2, 3], "evalu": 1, "rand": 1, "ensembled_explan": 1, "0": [1, 2, 3], "8": [1, 2, 3], "aggregating_func": [1, 3], "str": [1, 2, 3], "simplest": 1, "wai": 1, "provid": 1, "custom": [1, 3], "combin": 1, "get": 1, "one": [1, 2, 3], "string": 1, "avg": [1, 3], "min": [1, 3], "max": [1, 3], "captum": [1, 3], "attr": [1, 3], "integratedgradi": [1, 3], "gradientshap": 1, "salienc": 1, "net": 1, "imageclassifi": 1, "ig": 1, "attribut": [1, 3], "target": [1, 3], "sal": 1, "stack": [1, 2, 3], "dim": [1, 3], "agg": 1, "n_fold": [1, 3], "int": [1, 2], "shuffl": [1, 3], "fals": [1, 2, 3], "random_st": [1, 3], "train": [1, 3], "supervis": 1, "machin": 1, "learn": [1, 3], "sklearn": [1, 3], "kernel_ridg": 1, "kernelridg": [1, 3], "krr": [1, 3], "output": [1, 3], "y": 1, "k": 1, "fold": 1, "split": [1, 2, 3], "gener": [1, 3], "without": [1, 3], "inform": 1, "leakag": 1, "intern": 1, "model_select": 1, "kfold": 1, "make": [1, 4], "same": [1, 2], "shape": [1, 2, 3], "number": [1, 2], "greater": [1, 2], "than": [1, 2], "equal": [1, 2], "leav": 1, "out": 1, "done": [1, 2], "np": [1, 3], "sampl": [1, 3], "If": [1, 2], "uniform": 1, "auto": [1, 3], "invers": 1, "proport": 1, "area": [1, 2, 3], "also": 1, "arrai": [1, 3], "promot": 1, "smaller": 1, "true": [1, 2, 3], "onli": [1, 2], "which": [1, 2, 3], "l": [1, 2], "zou": [1, 2], "et": [1, 2], "al": [1, 2], "explain": [1, 2], "ai": [1, 2], "xai": [1, 2, 3], "sever": [1, 2], "commun": [1, 2], "acquir": [1, 2], "pneumonia": [1, 2], "covid": [1, 2], "19": [1, 2], "respiratori": [1, 2], "infect": [1, 2], "ieee": [1, 2], "transact": [1, 2], "artifici": [1, 2], "intellig": [1, 2], "1109": [1, 2], "tai": [1, 2], "2022": [1, 2], "3153754": [1, 2], "randint": 1, "low": 1, "high": 1, "krr_explan": 1, "threshold": [2, 3], "f1": [2, 3], "score": [2, 3], "recal": [2, 3], "precis": [2, 3], "harmon": [2, 3], "rang": [2, 3], "scenario": 2, "critic": [2, 3], "perfectli": 2, "match": 2, "propos": 2, "n": [2, 3], "width": 2, "height": 2, "repres": 2, "correl": 2, "presenc": 2, "consid": [2, 3], "pair": 2, "how": 2, "much": 2, "ha": [2, 3], "cover": [2, 3], "cross_2d": 2, "plus_2d": 2, "repeat": [2, 3], "b": 2, "20000001788139343": 2, "data": 2, "similar": 2, "classif": 2, "task": 2, "whole": 2, "over": [2, 3], "found": [2, 3], "intersect": [2, 3], "two": 2, "2000": 2, "images_tensor": 2, "predictor": 2, "explanation_threshold": 2, "replace_valu": 2, "compare_to": [2, 3], "same_predict": [2, 3], "chang": [2, 3], "probabl": [2, 3], "after": [2, 3], "hide": 2, "taken": 2, "account": 2, "ones": 2, "correspod": 2, "class": [2, 3], "predict": [2, 3], "origin": [2, 3], "Then": [2, 4], "best": 2, "all": 2, "meanwhil": 2, "situat": 2, "wa": 2, "close": 2, "obscur": [2, 3], "new_predict": [2, 3], "maxim": [2, 3], "irrelev": 2, "therefor": [2, 3], "modifi": 2, "On": 2, "other": 2, "hand": 2, "For": [2, 3], "while": 2, "differ": 2, "usag": [2, 4, 5], "alwai": 2, "The": 2, "stand": 2, "typic": 2, "case": 2, "possibli": 2, "wrap": 2, "nn": [2, 3], "softmax": [2, 3], "minim": 2, "replac": 2, "decid": 2, "whether": 2, "maximum": 2, "new": 2, "index": 2, "probability_origin": 2, "probability_hidden_area": 2, "_impact_ratio_help": 2, "wrapper": 2, "5": [2, 3], "zero": 2, "ex_explan": 2, "booltensor": 2, "def": [2, 3], "input_tensor": 2, "item": [2, 3], "val": 2, "els": [2, 3], "6": [2, 3], "19999998807907104": 2, "6000000238418579": 2, "photo": [2, 3], "do": 2, "diverg": 2, "ident": 2, "greatli": 2, "requir": [2, 3, 4], "depth": 2, "rgb": 2, "most": [2, 3], "stabl": 2, "first": [2, 3], "norm": 2, "matric": 2, "half": [2, 3], "18761281669139862": 2, "ones2": 2, "critica": 2, "metrics_scor": 2, "sum": 2, "time": 2, "13": 2, "11": 2, "tensor1": 2, "tensor2": 2, "threshold1": 2, "threshold2": 2, "absolute_valu": 2, "bool": 2, "logic": 2, "absolut": 2, "second": [2, 3], "boolean": 2, "divis": 2, "cross_2d_smal": 2, "plus_2d_smal": 2, "7": [2, 3], "intersection_area": 2, "union_area": 2, "3333333432674408": [2, 3], "matrix1": 2, "matrix2": 2, "sum_dim": 2, "By": 2, "work": 2, "last": [2, 3], "extend": 2, "remain": 2, "either": 2, "matrix": 2, "omit": 2, "both": [2, 3], "posit": [2, 3], "neg": [2, 3], "except": 2, "remov": 2, "specifi": 2, "onez_2d": 2, "zeroz_2d": 2, "onez_3d": 2, "zeroz_3d": 2, "4495": 2, "8990": 2, "onez_4d": 2, "zeroz_4d": 2, "4772": 2, "replacement_index": 2, "spot": 2, "4d": 2, "3d": 2, "given": 2, "along": 2, "fit": [2, 3], "image_4d": 2, "index_3d": 2, "replaced_imag": 2, "images_to_compar": 2, "epsilon": 2, "500": 2, "kwarg": 2, "between": 2, "As": 2, "creat": 2, "enough": 2, "some": 2, "here": [2, 3], "mai": 2, "take": 2, "signific": 2, "amount": 2, "process": 2, "power": 2, "memori": 2, "obtain": 2, "write": 2, "handl": 2, "might": 2, "additionali": 2, "choic": 2, "carefulli": 2, "test": [2, 3], "distanc": 2, "manual": 2, "recommend": [2, 4], "9": [2, 3], "explain_dummi": 2, "100": 2, "13370312750339508": 2, "length": 2, "product": 2, "result": [2, 3], "reduc": 2, "factor": 2, "start": 2, "thu": 2, "cannot": [2, 3], "larger": 2, "A": 2, "dim1": 2, "dim2": 2, "stacked_tensor": 2, "ensemblexai": [3, 4], "pretrain": 3, "resnet50": 3, "gc": 3, "o": 3, "json": 3, "pil": 3, "urllib": 3, "request": 3, "matplotlib": 3, "pyplot": 3, "plt": 3, "color": 3, "linearsegmentedcolormap": 3, "torchvis": 3, "transform": 3, "f": 3, "resiz": 3, "centercrop": 3, "resnet50_weight": 3, "occlus": 3, "noisetunnel": 3, "visual": 3, "viz": 3, "find": 3, "proper": 3, "name": 3, "id": 3, "under": 3, "link": 3, "urlopen": 3, "s3": 3, "amazonaw": 3, "com": [3, 4], "deep": 3, "imagenet_class_index": 3, "url": 3, "imagenet_classes_dict": 3, "helper": 3, "hasten": 3, "images_list": 3, "image_path": 3, "_crop": 3, "224": 3, "forward": 3, "_resiz": 3, "232": 3, "image_nam": 3, "listdir": 3, "open": 3, "append": 3, "load_al": 3, "classid": 3, "all_img": 3, "images_dir": 3, "all_img_org": 3, "all_ten": 3, "to_tensor": 3, "img": 3, "all_msk": 3, "masks_dir": 3, "tens_img": 3, "tens_msk": 3, "set": 3, "up": 3, "correct": 3, "path": 3, "imagenets50": 3, "lusseg": 3, "github": [3, 4], "io": 3, "pixel": 3, "perfect": 3, "50": 3, "sake": 3, "easier": 3, "n01491361": 3, "tiger": 3, "shark": 3, "input_dir": 3, "join": 3, "getcwd": 3, "sep": 3, "semi": 3, "segment": 3, "notebook": [3, 4], "all_imag": 3, "all_images_origin": 3, "all_tensor": 3, "all_mask": 3, "tensor_imag": 3, "tensor_mask": 3, "zip": 3, "cat": 3, "displai": 3, "to_pil_imag": 3, "our": 3, "three": 3, "eval": 3, "resnet_transform": 3, "pipelin": 3, "lambda": 3, "proper_data": 3, "_": 3, "pred": 3, "probs2": 3, "single_pr": 3, "unsqueez": 3, "single_data": 3, "integrated_gradi": 3, "attributions_ig": 3, "n_step": 3, "200": 3, "collect": 3, "transformed_img": 3, "default_cmap": 3, "from_list": 3, "blue": 3, "ffffff": 3, "25": 3, "000000": 3, "256": 3, "_1": 3, "visualize_image_attr_multipl": 3, "transpos": 3, "squeez": 3, "heat_map": 3, "original_imag": 3, "cmap": 3, "show_colorbar": 3, "outlier_perc": 3, "noise_tunnel": 3, "attributions_ig_nt": 3, "nt_sampl": 3, "nt_type": 3, "smoothgrad_sq": 3, "_2": 3, "attributions_occ": 3, "stride": 3, "sliding_window_shap": 3, "baselin": 3, "_3": 3, "cpu": 3, "detach": 3, "attributions_occ2": 3, "20": 3, "_4": 3, "attributions_occ_all_25": 3, "attributions_occ_all_15": 3, "attributions_ig_nt_al": 3, "across": 3, "53": 3, "03428497165441513": 3, "decis": 3, "impact": 3, "ratio": 3, "occlusion25": 3, "33": 3, "third": 3, "30": 3, "confid": 3, "impli": 3, "increas": 3, "004259963985532522": 3, "004277209285646677": 3, "accord": 3, "16": 3, "3734": 3, "5004": 3, "2527": 3, "17": 3, "0423": 3, "1679": 3, "1560": 3, "18": 3, "1734609603881836": 3, "iou": 3, "occlusion15": 3, "34": 3, "0009072763496078551": 3, "sample_xai": 3, "40": 3, "48": 3, "all_stack": 3, "aggreg_all1": 3, "aggreg_all2": 3, "aggreg_all3": 3, "explanations_al": 3, "01864352449774742": 3, "03320881724357605": 3, "04660182073712349": 3, "49": 3, "015071449801325798": 3, "022995954379439354": 3, "03292311355471611": 3, "83": 3, "valueerror": 3, "traceback": 3, "recent": 3, "call": 3, "cell": 3, "line": 3, "gt": 3, "39": 3, "file": 3, "mini": 3, "inzynierka": 3, "xai_ensemblings_bs_m": [3, 4], "py": 3, "367": 3, "361": 3, "362": 3, "alpha": 3, "regular": 3, "363": 3, "polynomi": 3, "choos": 3, "364": 3, "scikit": 3, "blob": 3, "main": 3, "pairwis": 3, "l2050": 3, "365": 3, "366": 3, "x_train": 3, "y_train": 3, "sample_weight": 3, "iter_weight": 3, "current": [3, 4], "group": 3, "368": 3, "y_predict": 3, "x_test": 3, "369": 3, "reshap": 3, "save": 3, "them": 3, "indic": 3, "recreat": 3, "order": 3, "later": 3, "1354752": 3, "66": 3, "451584": 3, "81": 3, "54": 3, "style": 3, "fast": 3, "plot_explan": 3, "columns_nam": 3, "classes_predict": 3, "method": 3, "nrow": 3, "ncol": 3, "len": 3, "fig": 3, "ax": 3, "subplot": 3, "figsiz": 3, "14": 3, "col": 3, "col_nam": 3, "titl": 3, "set_text": 3, "enumer": 3, "xaxi": 3, "set_ticks_posit": 3, "yaxi": 3, "set_yticklabel": 3, "set_xticklabel": 3, "imshow": 3, "vmin": 3, "vmax": 3, "255": 3, "set_ylabel": 3, "larg": 3, "expl": 3, "sign": 3, "amin": 3, "visualize_image_attr": 3, "plt_fig_axi": 3, "use_pyplot": 3, "show": 3, "47": 3, "predicted_nam": 3, "gradient": 3, "blended_heat_map": 3, "29": 3, "virtual": 4, "environ": 4, "strongli": 4, "encourag": 4, "clone": 4, "repositori": 4, "commandlin": 4, "activ": 4, "applic": 4, "pip": 4, "r": 4, "correct_path": 4, "txt": 4, "download": 4, "git": 4, "matts0000": 4, "egg": 4, "subdirectori": 4, "unavail": 4, "due": 4, "being": 4, "privat": 4, "To": 4, "run": 4, "document": 4, "creation": 4, "nbsphinx": 4, "extens": 4, "convert": 4, "html": 4, "seper": 4, "pandoc": 4, "conda": 4, "sphinx": 4, "sphinx_rtd_them": 4, "clean": 4, "doc": 4, "project": 4, "directori": 4, "user": 5, "instal": 5, "packag": 5, "develop": 5, "addit": 5, "imagenet": 5, "dataset": 5, "modul": 5}, "objects": {"": [[5, 0, 0, "-", "EnsembleXAI"]], "EnsembleXAI": [[1, 0, 0, "-", "Ensemble"], [2, 0, 0, "-", "Metrics"]], "EnsembleXAI.Ensemble": [[1, 1, 1, "", "autoweighted"], [1, 1, 1, "", "basic"], [1, 1, 1, "", "supervisedXAI"]], "EnsembleXAI.Metrics": [[2, 1, 1, "", "F1_score"], [2, 1, 1, "", "accordance_precision"], [2, 1, 1, "", "accordance_recall"], [2, 1, 1, "", "confidence_impact_ratio"], [2, 1, 1, "", "consistency"], [2, 1, 1, "", "decision_impact_ratio"], [2, 1, 1, "", "ensemble_score"], [2, 1, 1, "", "intersection_mask"], [2, 1, 1, "", "intersection_over_union"], [2, 1, 1, "", "matrix_2_norm"], [2, 1, 1, "", "replace_masks"], [2, 1, 1, "", "stability"], [2, 1, 1, "", "tensor_to_list_tensors"], [2, 1, 1, "", "union_mask"]]}, "objtypes": {"0": "py:module", "1": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"]}, "titleterms": {"ensemblexai": [0, 1, 2, 5], "packag": [0, 3, 4], "submodul": [0, 5], "modul": [0, 1, 2], "content": 0, "ensembl": [1, 3], "metric": [2, 3], "exampl": 3, "usag": 3, "imagenet": 3, "dataset": 3, "import": 3, "necessari": 3, "load": 3, "imag": 3, "plot": 3, "model": 3, "singl": 3, "explan": 3, "all": 3, "3": 3, "creat": 3, "user": 4, "instal": 4, "develop": 4, "addit": 4, "welcom": 5, "": 5, "document": 5, "get": 5, "start": 5, "index": 5}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"EnsembleXAI package": [[0, "ensemblexai-package"]], "Submodules": [[0, "submodules"]], "Module contents": [[0, "module-contents"]], "EnsembleXAI.Ensemble module": [[1, "module-EnsembleXAI.Ensemble"]], "EnsembleXAI.Metrics module": [[2, "module-EnsembleXAI.Metrics"]], "Example Usage on ImageNet dataset": [[3, "Example-Usage-on-ImageNet-dataset"]], "Imports of the necessary packages": [[3, "Imports-of-the-necessary-packages"]], "Loading images and plotting the examples": [[3, "Loading-images-and-plotting-the-examples"]], "Model Loading": [[3, "Model-Loading"]], "Single Explanations": [[3, "Single-Explanations"]], "Explanations for all 3 images": [[3, "Explanations-for-all-3-images"]], "Metrics for the created explanations": [[3, "Metrics-for-the-created-explanations"]], "Ensembles": [[3, "Ensembles"]], "Plotting all explanations": [[3, "Plotting-all-explanations"]], "User Installation": [[4, "user-installation"]], "Package Development Additional Installation": [[4, "package-development-additional-installation"]], "Welcome to EnsembleXAI\u2019s documentation!": [[5, "welcome-to-ensemblexai-s-documentation"]], "Getting Started:": [[5, null]], "Submodules:": [[5, null]], "Index": [[5, "index"]]}, "indexentries": {"ensemblexai.ensemble": [[1, "module-EnsembleXAI.Ensemble"]], "autoweighted() (in module ensemblexai.ensemble)": [[1, "EnsembleXAI.Ensemble.autoweighted"]], "basic() (in module ensemblexai.ensemble)": [[1, "EnsembleXAI.Ensemble.basic"]], "module": [[1, "module-EnsembleXAI.Ensemble"], [2, "module-EnsembleXAI.Metrics"], [5, "module-EnsembleXAI"]], "supervisedxai() (in module ensemblexai.ensemble)": [[1, "EnsembleXAI.Ensemble.supervisedXAI"]], "ensemblexai.metrics": [[2, "module-EnsembleXAI.Metrics"]], "f1_score() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.F1_score"]], "accordance_precision() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.accordance_precision"]], "accordance_recall() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.accordance_recall"]], "confidence_impact_ratio() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.confidence_impact_ratio"]], "consistency() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.consistency"]], "decision_impact_ratio() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.decision_impact_ratio"]], "ensemble_score() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.ensemble_score"]], "intersection_mask() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.intersection_mask"]], "intersection_over_union() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.intersection_over_union"]], "matrix_2_norm() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.matrix_2_norm"]], "replace_masks() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.replace_masks"]], "stability() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.stability"]], "tensor_to_list_tensors() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.tensor_to_list_tensors"]], "union_mask() (in module ensemblexai.metrics)": [[2, "EnsembleXAI.Metrics.union_mask"]], "ensemblexai": [[5, "module-EnsembleXAI"]]}})