from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Parser")
    parser.add_argument("path_to_json", help="Path to the json", type=str)
    args = parser.parse_args()

    