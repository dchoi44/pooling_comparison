import torch
import argparse
import logging
import os
import pathlib

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from sentence_transformers import losses, models, SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pooling', type=str, help='pooling method: one of [mean, max, cls]')
    parser.add_argument('--gpu', typr=int, help='specify gpu number')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    assert args.pooling in {'mean', 'max', 'cls'}, \
        f"{args.pooling}-pooling not supported. choose between mean, max, cls"

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    #### Download nfcorpus.zip dataset and unzip the dataset
    dataset = "msmarco"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
    #### Please Note not all datasets contain a dev split, comment out the line if such the case
    # dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

    #### Provide any sentence-transformers or HF model
    model_name = "bert-base-uncased"
    word_embedding_model = models.Transformer(model_name, max_seq_length=350)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #### Or provide pretrained sentence-transformer model
    # model = SentenceTransformer("msmarco-distilbert-base-v3")

    retriever = TrainRetriever(model=model, batch_size=32)

    #### Prepare training samples
    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    #### Training SBERT with cosine-product
    # train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    #### training SBERT with dot-product
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

    #### Prepare dev evaluator
    # ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

    #### If no dev set is present from above use dummy evaluator
    ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output",
                                   "{}-v1-{}-{}".format(model_name, dataset, args.pooling))
    os.makedirs(model_save_path, exist_ok=True)

    #### Configure Train params
    num_epochs = 25
    evaluation_steps = 10000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=ir_evaluator,
                  epochs=num_epochs,
                  output_path=model_save_path,
                  warmup_steps=warmup_steps,
                  evaluation_steps=evaluation_steps,
                  use_amp=True)


if __name__ == '__main__':
    main()
