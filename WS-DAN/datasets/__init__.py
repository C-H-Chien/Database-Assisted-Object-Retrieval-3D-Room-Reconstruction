from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .dog_dataset import DogDataset
from .furniture_bin_dataset import FurnitureBinDataset
from .furniture_sofa_dataset import FurnitureSofaDataset
from .furniture_chair_dataset import FurnitureChairDataset
from .furniture_table_dataset import FurnitureTableDataset
from .furniture_bed_dataset import FurnitureBedDataset
from .furniture_bookcase_dataset import FurnitureBookcaseDataset
from .furniture_cabinet_dataset import FurnitureCabinetDataset
from .furniture_display_dataset import FurnitureDisplayDataset

def get_trainval_datasets(tag, resize):
    if tag == 'aircraft':
        return AircraftDataset(phase='train', resize=resize), AircraftDataset(phase='val', resize=resize)
    elif tag == 'bird':
        return BirdDataset(phase='train', resize=resize), BirdDataset(phase='val', resize=resize)
    elif tag == 'car':
        return CarDataset(phase='train', resize=resize), CarDataset(phase='val', resize=resize)
    elif tag == 'dog':
        return DogDataset(phase='train', resize=resize), DogDataset(phase='val', resize=resize)
    elif tag == 'furniture_bin':
        return FurnitureBinDataset(phase='train', resize=resize), FurnitureBinDataset(phase='val', resize=resize)
    elif tag == 'furniture_sofa':
        return FurnitureSofaDataset(phase='train', resize=resize), FurnitureSofaDataset(phase='val', resize=resize)
    elif tag == 'furniture_chair':
        return FurnitureChairDataset(phase='train', resize=resize), FurnitureChairDataset(phase='val', resize=resize)
    elif tag == 'furniture_table':
        return FurnitureTableDataset(phase='train', resize=resize), FurnitureTableDataset(phase='val', resize=resize)
    elif tag == 'furniture_bed':
        return FurnitureBedDataset(phase='train', resize=resize), FurnitureBedDataset(phase='val', resize=resize)
    elif tag == 'furniture_bookcase':
        return FurnitureBookcaseDataset(phase='train', resize=resize), FurnitureBookcaseDataset(phase='val', resize=resize)
    elif tag == 'furniture_cabinet':
        return FurnitureCabinetDataset(phase='train', resize=resize), FurnitureCabinetDataset(phase='val', resize=resize)
    elif tag == 'furniture_display':
        return FurnitureDisplayDataset(phase='train', resize=resize), FurnitureDisplayDataset(phase='val', resize=resize)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))