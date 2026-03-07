import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_loader import get_model


def main():

    model = get_model("resnet18", 10)

    print(model)


if __name__ == "__main__":
    main()
