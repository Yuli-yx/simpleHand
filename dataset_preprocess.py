import numpy
import json




class Preprocess:
    """the annotation json file is formatted as:

    dict(
        xyz: List(np.array) # 21x3 
        uv: List(np.array) # 21x2 
        K: List(np.array) # 3x3 
        vertices: List(np.array) # 778x3
        image_path: string # *.jpg
    )"""
    def __init__(self, dataset='FPHAB', data_dir='dataset/train.json'):
        self.dataset = dataset
        self.data_dir = data_dir
        
    def process(self):
        passcde
    
    def save(self):
        pass
    

class FPHAB(Preprocess):
    def __init__(self, dataset='FPHAB', data_dir='dataset'):
        super().__init__(dataset, data_dir)
        self.train = ['Subject_1', 'Subject_2', 'Subject_3', 'Subject_4']
        self.eval = ['Subject_5', 'Subject_6']
    
    def get_info(self, image_path):
        train_info = []
        eval_info = []
        
        
    def process(self):
        with open(self.data_dir) as f:
            all_image_info = json.load(f)
        all_info = []
        for image_path in tqdm(all_image_info):
            info = read_info(image_path)
            info['image_path'] = image_path
            all_info.append(info)
        return all_info


if __name__ == "__main__":
    # preprocess FPHA dataset
    fpha = FPHAB()