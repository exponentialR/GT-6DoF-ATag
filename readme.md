## Generating April Tags
- To generate contiguous April Tags, you can use the `gen_april_tags` module. Below is an example command to generate tags with specific parameters:
  >   ````bash
  > python -m gen_april_tags --tag-size-mm 80 --id-start 46 --id-end 49 --pil-text
    
- To generate non-contiguous April Tags, you can use the:

    > ````bash
    > python -m src.gen_april_tags --tag-size-mm 40 --ids=19,20,21,22,27,28,29,30