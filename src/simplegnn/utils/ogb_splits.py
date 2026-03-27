def ogb_splits(output_path, db_name, *args, **kwargs):
    dataset_ogb = PygGraphPropPredDataset(name=db_name, root='tmp/')
    split_idx = dataset_ogb.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    return splits_from_index_lists([train_idx.tolist()], [valid_idx.tolist()], [test_idx.tolist()], db_name, output_path)
