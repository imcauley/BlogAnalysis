import BlogData

def average_length(corpus):
    total_lengths = 0

    for text,_ in corpus:
        total_lengths += len(text)

    return total_lengths / len(corpus)


if __name__ == '__main__':
    corpus = BlogData.get_data()

    print(average_length(corpus))