class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True, cfg=None):
        self.transform = transform
        self.relabel = relabel
        self.ori_data = copy.deepcopy( img_items )
        self.pid_dict = {}
        self.cfg = cfg
        if self.cfg:
            self.hpaug = HPAUG(cfg)
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                pid = self.get_pids(item[0], item[1])
                self.img_items.append((item[0], pid, item[2]))  # replace pid
                pids.add(pid)
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            print('len(pid)', len(self.pid_dict))
        else:
            self.img_items = img_items

        print('len(items)', len(self.img_items) )
    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        # print('idx', index)
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)

        if self.cfg:
            img = self.hpaug(img_path, img)
        ori_img = img
        if self.transform is not None: img = self.transform(img)
        img2 = ori_img.copy()
        if self.transform is not None: img2 = self.transform(img2)
        if self.relabel: pid = self.pid_dict[pid]

        return {
            'images': img,
            'targets': pid,
            'camid': camid,
            'img_path': img_path,
            'images2': img2
        }

    @staticmethod
    def get_pids(file_path, pid):
        """ Suitable for muilti-dataset training """
        return pid

    def update_pid_dict(self, pid_dict):
        self.pid_dict = pid_dict