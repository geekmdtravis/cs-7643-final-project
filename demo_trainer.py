from src.models import CXRModel
from src.utils import train_model


def main():
    model = CXRModel(model="densenet121_mm", freeze_backbone=True, hidden_dims=())
    train_model(model=model)


if __name__ == "__main__":
    main()
