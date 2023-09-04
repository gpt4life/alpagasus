import json
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_score(review):
    try:
        score = float(review.split('\n')[0])
    except Exception as e:          
        logger.error(
                f"{e}\nContent: {review}\n" "You must manually fix the score pair."
            )
        score = -1
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process the review and get scores.")
    parser.add_argument("-o", "--output-file")
    parser.add_argument(
        "--file",
        type=str,
        help='Input the review files'        
    )
    args = parser.parse_args()
    ratings = []
    data = json.load(open(args.file))
    for item in data:
        item['score'] = parse_score(item['review'])
        ratings.append(item)

    with open(f"{args.output_file}", "w") as output_review_file:
        json.dump(ratings, output_review_file, indent=4)
    