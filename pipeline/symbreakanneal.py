import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
import pickle
import os
from PIL import Image
import sklearn
from pipeline.symbreak import SymBreak

class SymBreakAnneal(SymBreak):
    def __init__(self,
        data_path = 'data/smaller_example_coords.csv',
        save_path = "results/",
        save_plots = True,
        pad = 700,
        org_rad = 75,
        org_pad = 25,
        snapshot_step = 100,
        min_dist = 75 + 10,
        c_to_c_dist = 2*75+2*1,
        objective = "min",
        lmbda = None,
        # seq gen configs
        n_init_orgs = 1, 
        n_total_orgs = 6,
        n_search = 3,
        sample_mask_size = 2000, 
        # sim anneal configs
        niter = 2000,
        move_len = 80,
        move_decay = .9991,
        random_perturb = 1/5,
        perturb_decay = .9977,
        model_path = "models/krr_model.pkl",
        scaler_path = "models/scaler.pkl"
    ):
        super().__init__(
            data_path = data_path,
            save_path = save_path,
            save_plots = save_plots,
            pad = pad,
            org_rad = org_rad,
            model_path = model_path,
            scaler_path = scaler_path
        )
        self.data_path = data_path
        self.save_path = save_path
        self.save_plots = save_plots
        self.pad = pad
        self.org_rad = org_rad
        self.org_pad = org_pad
        self.snapshot_step = snapshot_step
        self.min_dist = min_dist
        self.c_to_c_dist = c_to_c_dist
        self.objective = objective
        self.lmbda = lmbda
        # seq gen configs
        self.n_init_orgs = n_init_orgs 
        self.n_total_orgs = n_total_orgs
        self.n_search = n_search
        self.sample_mask_size = sample_mask_size 
        # sim anneal configs
        self.niter = niter
        self.move_len = move_len
        self.move_decay = move_decay
        self.random_perturb = random_perturb
        self.perturb_decay = perturb_decay
        self.model_path = model_path
        self.scaler_path = scaler_path

        loaded_model = pickle.load(open(self.model_path, 'rb'))
        loaded_scaler = pickle.load(open(self.scaler_path,'rb'))

        self.model = loaded_model
        self.scaler = loaded_scaler

        centroids = pd.read_csv(self.data_path, header=None).values.astype(int)
        self.centroids = self.shift(centroids)

    def sim_anneal(self, im, centroids):
        #print("Starting simulated annealing...")
        log = []

        curr_score, preds, feats = self.evaluate(im, centroids)
        log.append([preds, feats])

        for i in tqdm(range(self.niter)):
            # get candidate perturbation
            new_im, new_centroids, sim_score, preds, newfeats = self.candidate(im, centroids, i)
            
            # determine random acceptance threshold
            p = np.random.uniform(0,1)
            p_perturb = self.random_perturb*self.perturb_decay**(i+1)

            # accept if good score or if passes threshold
            if  sim_score > curr_score or p < p_perturb:
                im, centroids, curr_score, feats = new_im, new_centroids, sim_score, newfeats

            log.append([preds, feats])
            #wandb.log({"curr_score": curr_score})
            if i % self.snapshot_step == 0:
                name = "anneal_step_%d" % i
                #wandb.log({name: wandb.Image(im.astype(np.uint8))})
        return im, centroids, log

    def candidate(self, image, centers, ix):
        test_img = image.copy()
        test_centers = centers.copy()
        im_size = len(image)
        assert(self.sample_mask_size < im_size)
        
        nx, ny, ix = self.random_move(test_centers, im_size, ix)

        old_x, old_y = test_centers[ix]
        # erase
        test_img = self.circle(old_x, old_y, test_img, self.org_rad, fill=True, fillVal=0)
        # fill new
        test_img = self.circle(nx, ny, test_img, self.org_rad, fill=True)
        #test_img[old_x-75:old_x+75, old_y-75:old_y+75] = 50
        #test_img[nx-75:nx+75, ny-75:ny+75] = 255

        test_centers[ix] = (nx, ny)
        sim_score, preds, newfeats = self.evaluate(test_img, test_centers)
        
        return test_img, test_centers, sim_score, preds, newfeats

    def random_move(self, centers, im_size, ix):
        found_valid_move = False
        newx, newy, random_index = 0, 0, 0
        
        while not found_valid_move:
            c_num = len(centers)
            cixs = list(range(c_num))
            
            # randomly choose organoid to be perturbed
            random_index = np.random.choice(cixs)

            # determine random x, y perturbation
            angle = np.pi * np.random.uniform(0, 2)
            length = self.move_len*self.move_decay**(ix+1)
            dx, dy = length*np.cos(angle), length*np.sin(angle)
            dx, dy = np.rint(dx), np.rint(dy)
            newx, newy = int(centers[random_index][0] + dx), int(centers[random_index][1] + dy)

            # get the neighbors of chosen organoid
            if random_index == 0:
                nbors = centers[1:]
            elif random_index == c_num-1:
                nbors = centers[:-1]
            else:
                nbors = centers[:random_index] + centers[random_index+1:]
            
            found_valid_move = self.validate(newx, newy, nbors, im_size)
        return newx, newy, random_index

    def validate(self, cx, cy, centroids, im_size):
        # check if out of bounds   
        cxbool = self.out_of_bounds_check(cx, im_size)
        cybool = self.out_of_bounds_check(cy, im_size)
        
        # check centroid to centroid distance
        rep = np.tile(np.array([cx, cy]).reshape(-1,2), [len(centroids), 1])
        centroids_arr = np.array(centroids)
        dist = np.sqrt(np.sum((rep-centroids_arr)**2, axis = 1))
        valid = np.all(dist > self.c_to_c_dist)
        
        check = cxbool and cybool and valid

        return check

    def out_of_bounds_check(self, coord, im_size):
        if (coord < self.min_dist) or (coord > im_size - self.min_dist):
            return False
        else:
            return True
        
    """ def get_weights(self, arr, weights):
        if weights == "uniform":
            n = len(arr)
            return [1/n]*len(arr)
        elif weights == "exponential":
            phiArr = np.exp(-1*arr)
            weights = phiArr / np.sum(phiArr)
        else:
            maxVal, minVal = np.max(arr), np.min(arr)
            phiArr = -1*arr+minVal+maxVal
            weights = phiArr / np.sum(phiArr)
            return weights """

    def evaluate(self, im_pattern, centroids):
        new_feats = []

        feats = self.extract_features(im_pattern, centroids, messages=False)

        feats_ixs = np.array([0,1,1,0]).astype(np.bool8)
        X = self.scaler.transform(feats)
        preds = self.model.predict(X)

        reward = self.objective_fn(preds, self.objective, self.lmbda)

        return reward, preds, X

    def objective_fn(self, preds, objective, lmbda = None):
        if objective == "min":
            return np.min(preds)
        elif objective == "mean":
            return np.mean(preds)
        else:
            assert(lmbda != None)
            return np.mean(preds) + lmbda*np.min(preds)
        

    def sample_pattern(self, save_samples=False):
        print("Sampling a pattern...")
        # create empty pattern
        mask = np.zeros((self.sample_mask_size, self.sample_mask_size))
        centroids = []

        # Initialize mask
        for _ in range(self.n_init_orgs):
            x, y = self.random_location(mask)
            centroids.append((x,y))
            mask = self.circle(x,y, mask, self.org_rad, fill=True)
        
        # Sequentially add organoids stochastically
        for _ in tqdm(range(self.n_total_orgs-1)):
            sim_organoids = []
            for _ in range(self.n_search):
                # copy to evaluate independently
                test_mask = mask.copy()
                test_cents = centroids.copy()

                # sample test location in mask
                x, y = self.random_location(mask)
                test_cents.append((x,y))
                test_mask = self.circle(x,y, test_mask, self.org_rad, fill=True)
                score, _, _ = self.evaluate(test_mask, test_cents)
                sim_organoids.append((score, x, y))
            
            # select best simulated organoid
            best_sim_organoid = sorted(sim_organoids)[-1]
            newx, newy = best_sim_organoid[1], best_sim_organoid[2]
            mask = self.circle(newx, newy, mask, self.org_rad, fill = True)
            centroids.append((newx, newy))

        # add the pad at the end
        xmin, xmax = np.min(centroids[:, 0]), np.max(centroids[:, 0])
        x_mean = (xmax-xmin)//2
        ymin, ymax = np.min(centroids[:, 1]), np.max(centroids[:, 1])
        y_mean = (ymax-ymin)//2

        im_mean = self.sample_mask_size//2
        x_shift = im_mean - x_mean
        y_shift = im_mean - y_mean

        gen_centroids = np.array(centroids) - np.array([x_shift, y_shift])
        if self.save_plots:
            plt.scatter(gen_centroids[:, 0], gen_centroids[:, 1])
            pth = os.path.join(self.save_path, "sampled_scatter.jpeg")
            plt.savefig(pth) 

        gen_pattern = np.pad(mask, self.pad)
        #gen_centroids = np.array(centroids) + self.pad

        if self.save_plots:
            plt.imshow(gen_pattern)
            pth = os.path.join(self.save_path, "sampled_pattern.jpeg")
            plt.savefig(pth) 

        if save_samples:
            gen_centroids



        return gen_centroids

    def random_location(self, image):
        found_valid = False
        
        while not found_valid:
            x, y = np.random.choice(self.sample_mask_size, 2)
            # ensure organoid is at least 2 pads worth away from wall and organoids
            if (x > self.min_dist and x < self.sample_mask_size-self.min_dist and 
                y > self.min_dist and y < self.sample_mask_size-self.min_dist):
                bool_mat = self.circle(x, y, image, self.min_dist)
                
                # check if overlapping with other organoids
                if np.sum(image[bool_mat]) == 0: 
                    found_valid = True
        return x,y

    