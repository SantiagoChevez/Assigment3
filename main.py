from embeddings import skip_gram_embeddings
import train


def run_skipgram_pipeline():
    print('\n1) Generating Skip-gram embeddings and dataset')
    # This will train Word2Vec (skip-gram), compute TF-IDF weighted doc vectors and
    # save `vectorized_news_skip_gram_embeddings.csv` in the repo root.
    skip_gram_embeddings()

    print('\n2) Training classifier on Skip-gram vectors')
    # train.train() will load `vectorized_news_skip_gram_embeddings.csv`, train the MLP,
    # print training/validation metrics and save `skipgram.pth`.
    train.train_skip()

    print('\nSkip-gram pipeline finished.')


def run_cbow_pipeline():
    # Placeholder for CBOW: implement CBOW training + dataset generation here later.
    print('CBOW pipeline placeholder — not implemented yet')


def main():
    run_skipgram_pipeline()
    # leave space to run CBOW in future
    # run_cbow_pipeline()


if __name__ == '__main__':
    main()
