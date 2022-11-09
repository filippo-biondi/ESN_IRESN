from sklearn.model_selection import KFold
from generators import Bayesian_hp_generator


def search_direct(model_builder, hp_generator, train_val_set, train_val_label, val_function, **kwargs):
    best_val = 0
    best_train = 0
    best_params = None
    best_model = None
    input_shape = (None, train_val_set.shape[-1])

    for i, hp in enumerate(hp_generator):
        print("random search n. ", i + 1, end='\r')
        full_hp = {k: v for k, v in hp.items()}
        full_hp["input_shape"] = input_shape
        full_hp["reg"] = None
        model = model_builder(full_hp)
        train, val, reg = val_function(model, train_val_set, train_val_label, **kwargs)
        full_hp["reg"] = reg
        if val > best_val:
            best_train = train
            best_val = val
            best_params = full_hp
            best_model = model
        if isinstance(hp_generator, Bayesian_hp_generator):
            hp_generator.tell(val)

    best_model.fit(train_val_set, train_val_label, reg=best_params["reg"])
    print("")
    return best_model, best_params, best_train, best_val


def k_fold_val_reg(model, x, y, **kwargs):
    n_fold = kwargs["n_fold"]
    kf = KFold(n_splits=n_fold)

    best_avg_val = 0
    best_avg_train = 0
    best_reg = 0
    states = model.computeStates(x).numpy()
    if model.lasso:
        reg_values = [0.001, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.00001, 0.0000005, 0.0000001]
    else:
        reg_values = [1, 0.1, 0.01, 0.001, 0.0001]
    for reg in reg_values:
        avg_val = 0
        avg_train = 0
        for train_index, val_index in kf.split(x):
            avg_train += model.trainReadout(states[train_index], y[train_index], reg=reg)
            avg_val += model.evaluateReadout(states[val_index], y[val_index])
        avg_val /= n_fold
        avg_train /= n_fold
        if avg_val > best_avg_val:
            best_avg_val = avg_val
            best_avg_train = avg_train
            best_reg = reg
    return best_avg_train, best_avg_val, best_reg


def double_k_fold(x, y, n_out_fold, n_in_fold, n_restart, model_builder, hp_generator, n_iter, search, val_fun,
                  **kwargs):
    kf_out = KFold(n_splits=n_out_fold)
    models = [[] for _ in range(n_out_fold)]
    i = 0
    for train_val_index, test_index in kf_out.split(x):
        print("out fold ", i + 1)
        for j in range(n_restart):
            print("restart ", j + 1)
            m, param, train, val = search(model_builder, hp_generator(n_iter=n_iter, **kwargs),
                                          x[train_val_index],
                                          y[train_val_index], val_fun, n_fold=n_in_fold)
            test = m.evaluate_precise(x[test_index], y[test_index])
            models[i].append((m, param, train, val, test))
        i += 1
    return models
