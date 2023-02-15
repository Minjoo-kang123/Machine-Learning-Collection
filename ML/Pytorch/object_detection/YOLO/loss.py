"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        #저희가 두 점 사이의 거리를 구하기 위해 사용했던 식을 생각하면 될 것 같습니다. 식은 다음과 같습니다. 여기서 output은 예측값, target은 정답값입니다.
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # 대상 bbox를 사용하여 예측된 두 바운딩 박스에 대한 IoU를 계산
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        #둘을 합치되 첫번째 차원을 합쳐서 늘려라
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # 두 예측 중 가장 높은 IoU를 가진 상자를 가져옴
        # best_box 변수에는 두 bounding box 중 IoU 값이 더 큰 box의 index가 저장됨
        # 베스트박스는 0, 1의 인덱스가 될 예정
        iou_maxes, bestbox = torch.max(ious, dim=0)
        #ground truth box의 중심이 존재하는지 여부를 확인합니다.만약 존재한다면 exists_box = 1, 존재하지 않는다면 exists_box = 0이 될
        exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i


        #   상자좌표    #

        # 객체가 없는 상자를 0으로 설정합니다. 두 개의 예측 중 하나만 제거합니다.
        # 예측 중 하나만 제거하는데, 이 예측은 이전에 계산된 Iou가 가장 높은 예측입니다.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        #1.Localization loss 이후 w,h에 루트를 씌운 뒤 MSE 계산
        # 상자의 너비, 높이의 제곱을 구하여 다음을 확인합니다.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box는 IoU가 가장 높은 bbox에 대한 신뢰도 점수입니다.
        #predictions[..., 25:26]: 첫번째 box의 confidence score
        #predictions[..., 20:21]: 두번째 box의 confidence score
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        #flatten > 1차원 함수로 만들어줌 해당은 1차원으로 평탄하 됨
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        #object가 없는 곳의 loss exitsts box는 존재할 시 1 아니면 0으로 해당 박스 내 존재의 유무를 판단해줌
        max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )


        #두박스 모두 참여 각 -2 즉 4차원중 123차원을 합치고 마지막 차원은 그대로 둔 채로 제곱 합을 한다.
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )
        print(class_loss)
        #최종 로스 계산
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
