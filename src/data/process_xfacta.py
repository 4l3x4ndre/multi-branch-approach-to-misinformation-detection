from tqdm import tqdm
import json
import os
from torch.utils.data import DataLoader
from loguru import logger

from corpus_truth_manipulation.config import CONFIG, XFACTA, XFACTA_FORMATTED
from corpus_truth_manipulation.dataset import MMFakeBenchDataset, collate_fn

def main():
    os.makedirs(XFACTA_FORMATTED, exist_ok=True)

    # ====================================================================
    # Step 1: Merge dev.json and test.json into one xfacta data set file.
    # ====================================================================
    dev_path = XFACTA / "dev.json"
    test_path = XFACTA / "test.json"
    # Merge the two files into one json:
    merged_data = []
    for file_path in [dev_path, test_path]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)
    merged_output_path = XFACTA / "xfacta.json"
    with open(merged_output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    # ====================================================================
    # Step 2: Reformat json to match MMFakeBench format.
    # ====================================================================
    formatted_data = []
    known_fake_classes = set()
    for record in tqdm(merged_data, desc="Formatting XFacta json"):
        xf_name = record['images'][0].split('hzy/')[1].split('/')[0] # XF of XFacta

        if 'https://' in record['images'][0]:# or 'https://' in record_entry['ooc_tweet']['images'][0]:
            continue

        formatted_record = {
            "text": record['text'],
            "gt_answers": "True" if record['label'] == True else "False",
            "image_path": record['images'][0].split(xf_name)[1],
        }
    
        record_entry_path_jpg = record['images'][0].split(f'{xf_name}/')[1] # real_sample/media/batch4/49/images/img0.jpeg"
        record_entry_path_rf = record_entry_path_jpg.split('/')[0] # 'real_sample'
        record_entry_path_batch = record_entry_path_jpg.split('media/')[1].split('/')[0] # 'batch4'
        record_entry_path = os.path.join(record_entry_path_rf, f'{record_entry_path_batch}.json')
        record_entry_id =  record_entry_path_jpg.split(f'{record_entry_path_batch}/')[1].split('/')[0] # 49
        with open(XFACTA / record_entry_path, 'r', encoding='utf-8') as f:
            record_entries = json.load(f)
        for entry in record_entries:
            if entry['tweet_id'] == str(record_entry_id):
                record_entry = entry
                break

        if record['label'] == False:
            fake_cls = record_entry['ooc_tweet']['metadata']['error_category']
            if isinstance(fake_cls, list):
                continue

            formatted_record['fake_cls'] = fake_cls
            formatted_record['image_source'] = 'fake'
            formatted_record['text_source'] = 'fake'
        else:
            formatted_record['fake_cls'] = 'original'
            formatted_record['image_source'] = record_entry['ooc_tweet']['metadata']['author_id']
            formatted_record['text_source'] = record_entry['ooc_tweet']['metadata']['author_id']
        formatted_data.append(formatted_record)

    os.makedirs(XFACTA_FORMATTED / "test", exist_ok=True)
    os.makedirs(XFACTA_FORMATTED / "test" / "source", exist_ok=True)
    formatted_output_path = XFACTA_FORMATTED / "test" / "source" / "xfacta.json"
    with open(formatted_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

    # Copy fake sample and real sample to XFACTA_FORMATTED if not exist
    for sample_type in ['fake_sample', 'real_sample']:
        src_dir = XFACTA / sample_type
        dest_dir = XFACTA_FORMATTED / "test" / sample_type
        if not os.path.exists(dest_dir):
            logger.info(f'Copying {sample_type} from {src_dir} to {dest_dir}...')
            os.system(f'cp -r {src_dir} {dest_dir}')


    # ====================================================================
    # Step 3: Extract graphs
    # ===================================================================
    CONFIG.device = 'cpu'
    dataset = MMFakeBenchDataset(
        data_root=XFACTA_FORMATTED,
        split='test',
        transform=None,
        include_img=False,
        shuffle_data=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.data.batch_size,
        shuffle=False,
        num_workers=CONFIG.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    logger.info(f"Length of XFacta formatted dataset: {len(dataset)}")




if __name__ == '__main__':
    main()
