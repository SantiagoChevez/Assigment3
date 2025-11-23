from embeddings import skip_gram_embeddings, cbow_embeddings
import train


def run_skipgram_pipeline():
    print('\n1) Generating Skip-gram embeddings and dataset')
    skip_gram_embeddings()

    print('\n2) Training classifier on Skip-gram vectors')
    train.train_skip()

    print('\nSkip-gram pipeline finished.')


def run_cbow_pipeline():
    print('1) Generating CBOW embeddings and dataset')
    cbow_embeddings()
    print('\n2) Training classifier on CBOW vectors')
    train.train_cbow()
    print('\nCBOW pipeline finished.')


def main():
    run_skipgram_pipeline()
    run_cbow_pipeline()


if __name__ == '__main__':
    main()