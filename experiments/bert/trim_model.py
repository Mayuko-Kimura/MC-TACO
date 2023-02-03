import argparse

import torch
from pytorch_pretrained_bert import BertForSequenceClassification


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="saved model path.")

    parser.add_argument("--target_model",
                        default=None,
                        type=str,
                        required=True,
                        help="target model.")

    parser.add_argument("--save_model",
                        default=None,
                        type=str,
                        required=True,
                        help="path of the model to save.")
    args = parser.parse_args()

    model_path = args.model_path

    state_dict = torch.load(model_path)

    base_model = BertForSequenceClassification.from_pretrained(args.target_model,num_labels=2)

    base_model_keys = []
    print(state_dict.keys())    

    for k,v in base_model.state_dict().items():
        base_model_keys.append(k)

    old_state_dict = {}
    for key, val in state_dict.items():
        print(key)
        prefix = key.split('.')[0]
        if prefix == 'scoring_list' or key not in base_model_keys or 'classifier.' in key:
            print(key)
            continue
        old_state_dict[key] = val

    state_dict = {'state': old_state_dict}
    torch.save(state_dict['state'], args.save_model)

if __name__ == "__main__":
    main()
