# Evidence Based Verification for Real World Information Needs
## License

This library is licensed under the Apache 2.0 License. Some portions of the code-base were based off the code for the original [https://github.com/awslabs/fever/tree/master/fever-annotations-platform](https://github.com/awslabs/fever) which is licensed under the Apache 2.0 License by Amazon Research Cambridge. 

## Note

This service is provided `as-is` for information purposes with no guarantees of fitness for purpose.

## Service

The annotation service is a flask service that runs from this entrypoint. It expects a mongodb with HITs and claims already loaded.

```
export PYTHONPATH=src
python src/annotation/annotation_service.py
```

## Loading data

The database can be populated with the scripts in `src/annotation/etl/`, such as `load_from_boolq_gold_3.py`, but these are largely undocumented and unmaintained.

The Wikipedia pages for BoolQ claims can be downloaded with `src/dataset/construction/download_wiki_pages.py`.
