# =============================================================================================
# Returning ShapeNetCore Database Model IDs According to the FGVC net Furniture Category Label
# -- Chiang-Heng Chien --
# =============================================================================================
import os
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

def model_ID_retrieval(furniture_class, furniture_category):
    if furniture_class == 1:
        # -- bed --
        if furniture_category == 1:
            return '30eb0a2f835b91b19e9c6b0794a3b4d8'
        elif furniture_category == 2:
            return 'c9eb8864f33152a3e09520ec2bf55884'
        elif furniture_category == 3:
            return 'af2e51ff34f1084564dabb9b8161e0a7'
        elif furniture_category == 4:
            return 'edf13191dacf07af42d7295fb0533ac0'
        elif furniture_category == 5:
            return '9c203361eac308754b8a0e1f781de28e'
        elif furniture_category == 6:
            return '1f11b3d9953fabcf8b4396b18c85cf0f'
        elif furniture_category == 7:
            return '6284bce6e6fe15c8aa69dfdc5532bb13'
        
    elif furniture_class == 2:
        # -- bin --
        if furniture_category == 0:
            return '2f1aed8925147120c62ac18334863d36'
        elif furniture_category == 1:
            return '4a14e1ffc27ba762ad5067eac75a07f7'
        elif furniture_category == 2:
            return '62f4ed6e1df63042cecaed25e0da0964'
        elif furniture_category == 3:
            return '93d5f21e94ac5fb68a606c63493ddb7c'
        elif furniture_category == 4:
            return '632c8c69e7e7bda54559a6e3650dcd3'
        elif furniture_category == 5:
            return 'af7a781f08fdd4481faebbdea6bd9be'
        elif furniture_category == 6:
            return 'da9ee9282a5e7c77ad5067eac75a07f7'
        elif furniture_category == 7:
            return 'e7de8a24dfc385d92715de3ea7b582d7'
        elif furniture_category == 8:
            return 'f8b6bcf0dcf240e567e4c36fcbad1d04'

    elif furniture_class == 3:
        # -- bookcase --
        if furniture_category == 1:
            return '586356e8809b6a678d44b95ca8abc7b2'
        elif furniture_category == 2:
            return '46579c6050cac50a1c8c7b57a94dbb2e'
        elif furniture_category == 3:
            return '3c9f48785776ee8efcd4910413c446d9'
        elif furniture_category == 4:
            return '4e26a2e39e3b6d39961b70a6f96df2a4'
        elif furniture_category == 5:
            return 'e7e3aacb7ef4ef00f11caac3db6956bf'
        elif furniture_category == 6:
            return '1b3090bfb11cf834f06824a9291300d4'
        elif furniture_category == 7:
            return '131b5b691ff5bff3945770bff82992ca'
        elif furniture_category == 8:
            return '33b29c9e55250cc57f0bdbbc607f1f0e'

    elif furniture_class == 4:
        # -- chair --
        if furniture_category == 1:
            return '1be38f2624022098f71e06115e9c3b3e'
        elif furniture_category == 2:
            return '2e8748c612c5d3519fb4103277a6b93'
        elif furniture_category == 3:
            return '3f4e88498f4e54993f7e27638e63d848'
        elif furniture_category == 4:
            return '4a9ac9e0b3a804bc8fb952c92850e1dc'
        elif furniture_category == 5:
            return '8ab6783b1dfbf3a8a5d9ad16964840ab'
        elif furniture_category == 6:
            return '791c488a167bd73f91663a74ccd2338'
        elif furniture_category == 7:
            return '9368cd9028151e1e9d51a07a5989d077'
        elif furniture_category == 8:
            return '9368cd9028151e1e9d51a07a5989d077' # -- dummy because no suitable cad model matches this type of chair --
        elif furniture_category == 9:
            return '33c4f94e97c3fefd19fb4103277a6b93'
        elif furniture_category == 10:
            return '51c276e96ac4c04ebe67d9b32c3ddf8'
        elif furniture_category == 11:
            return '1f82011c2303fc7babb8f41baefc4b12'

    elif furniture_class == 5:
        # -- cabinet --
        if furniture_category == 1:
            return 'dca4c8bdab8bdfe739e1d6694e046e01'
        elif furniture_category == 2:
            return 'b278c93f18480b362ea98d69e91ba870'
        elif furniture_category == 3:
            return '10c484963692d3c87d40405e5ee68b4f'
        elif furniture_category == 4:
            return 'b7dc1e9a3949991747d7c2aae1e5c61'
        elif furniture_category == 5:
            return '1b32d878eeb305e45588a2543ef0b0b4'
        
    elif furniture_class == 6:
        # -- display --
        if furniture_category == 1:
            return '6fcbee5937d1cf28e5dbcc9343304f4a'
        elif furniture_category == 2:
            return 'ab68830951b9a5a02ba94c8f6eceab22'
        elif furniture_category == 3:
            return 'b4386438057e85988bfba3df1f776207'
        elif furniture_category == 4:
            return 'dd8cebcb4d0564059352b002a7d38daa'
        elif furniture_category == 5:
            return 'fa7324388797a2b41143ab0877fec7c3'
        elif furniture_category == 6:
            return 'fa7324388797a2b41143ab0877fec7c3'# -- dummy id here... need to find proper cad model --

    elif furniture_class == 7:
        # -- sofa --
        if furniture_category == 1:
            return '117f6ac4bcd75d8b4ad65adb06bbae49'
        elif furniture_category == 2:
            return '849ddda40bd6540efac8371a83e130ac'
        elif furniture_category == 3:
            return '1230d31e3a6cbf309cd431573238602d'
        elif furniture_category == 4:
            return '511168d4461d169991a3d45e8887248a'
        elif furniture_category == 5:
            return '679010d35da8193219fb4103277a6b93'
        elif furniture_category == 6:
            return 'f20e7f4f41f323a04b3c42e318f3affc'
        elif furniture_category == 7:
            return 'f20e7f4f41f323a04b3c42e318f3affc'# -- dummy id here... need to find proper cad model --
        
    elif furniture_class == 8:
        # -- table --
        if furniture_category == 1:
            return '1d447e3b068b924ad91787f0eb159c8c'
        elif furniture_category == 2:
            return '2a0b5875dc999a7a29e047111bd79063'
        elif furniture_category == 3:
            return '3b0c62bde7b24de85ce578b5b4bfae3c'
        elif furniture_category == 4:
            return '4bbf789edb243cafc955e5ed03ef3a2f'
        elif furniture_category == 5:
            return '4c5bc4f3d5a37c6dca9d5f70cc5f6d22'
        elif furniture_category == 6:
            return '4d2f7c689e77df6b6dc1766995c17a41'
        elif furniture_category == 7:
            return '8d143c8169ed42ada6fee8e2140acec9'
        elif furniture_category == 8:
            return 'c6c412c771ab0ae015a34fa27bdf3d03'
        