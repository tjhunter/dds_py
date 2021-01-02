# Important: the preprocessors must run ordered
c.preprocessors=["nbconvert.preprocessors.ExecutePreprocessor","nbconvert.preprocessors.TagRemovePreprocessor"]
c.ClearMetadataPreprocessor.enabled = False
c.TagRemovePreprocessor.enabled = True
c.TagRemovePreprocessor.remove_input_tags = set(["hide_cell"])
