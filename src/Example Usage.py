#Example Usage
cat_cols_subset2 = ["pymnt_plan", "purpose", "title", "zip_code", "addr_state"]

transformer = Subset2CategoricalPerformanceTransformer(cat_cols=cat_cols_subset2)

transformer.fit(X_train[cat_cols_subset2])

X_train_cat = transformer.transform(X_train[cat_cols_subset2])
