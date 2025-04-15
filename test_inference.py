from inference import predict

support_imgs = [
    "dataset_split/train/n04509417/n04509417_405.jpeg",
    "dataset_split/train/n04509417/n04509417_1167.jpeg",
    "dataset_split/train/n04515003/n04515003_6.jpeg",
    "dataset_split/train/n04515003/n04515003_713.jpeg"
]

support_lbls = [0, 0, 1, 1]

query_img = "test_images/image5.jpeg"

print("Prediction:", predict(support_imgs, support_lbls, query_img))
