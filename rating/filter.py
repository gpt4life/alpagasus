import json
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_score(review):
    try:
        score = float(review.split('\n')[0])
    except Exception as e:
        if('score:' in review):
            score = float(review.split('score:')[1].split('\n')[0])
        elif('Score:' in review):
            score = float(review.split('Score:')[1].strip('\n')[0])
        else:           
            logger.error(
                f"{e}\nContent: {review}\n" "You must manually fix the score pair."
            )
            score = -1
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter low-quality data.")
    parser.add_argument("-o", "--output-file")
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=4.5,
        help="The threshold for filtering."
    )
    parser.add_argument(
        "--file_a",
        type=str,
        help='the original alpaca(the original data) file'        
    )
    parser.add_argument(
        '--file_b',
        type=str,
        help='the rating files'
    )
    args = parser.parse_args()
    alpaca = json.load(open(args.file_a))
    ratings = json.load(open(args.file_b))
    T = args.threshold

    filtered = []
    for i in range(len(alpaca)):
        if ratings[i]['score'] >= T:
            filtered.append(alpaca[i])
    

    with open(f"{args.output_file}", "w") as output_review_file:
        json.dump(filtered, output_review_file, indent=4)
        
        
    
    
    