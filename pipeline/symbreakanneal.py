import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import sklearn
from pipeline.symbreak import SymBreak

class SymBreakAnneal(SymBreak):
    def __init__(self,
        data_path = None,
        save_path = "results/",
        save_plots = True,
        pad = 700,
        org_rad = 75,
        org_pad = 25,
        snapshot_step = 100,
        min_dist = 75 + 5,
        c_to_c_dist = 2*75+2*1,
        objective = "min",
        lmbda = None,
        # seq gen configs
        n_init_orgs = 1, 
        n_total_orgs = 6,
        n_search = 10,
        mask_size = 1300, 
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
        self.mask_size = mask_size 
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

        self.centroids = None
        if data_path:
            centroids = pd.read_csv(self.data_path, header=None).values.astype(int)
            self.centroids = self.shift(centroids).tolist()
            self.mask_size = np.max(centroids) + self.pad

    def sim_anneal(self, centroids = None):
        if centroids: # sample has been generated
            self.centroids = centroids.tolist()
            self.mask_size = np.max(centroids) + self.pad
        else: # coordinates have been provided in class init
            assert(self.centroids is not None)
            centroids = self.centroids

        print("Annealing...")
        #print("Starting simulated annealing...")
        
        #mask = self.make_mask(centroids)
        curr_score, preds, feats = self.evaluate(centroids)

        for t in tqdm(range(self.niter)):
            # get candidate perturbation
            new_centroids, sim_score, preds, newfeats = self.candidate(centroids, t)
            
            # determine random acceptance threshold
            p = np.random.uniform(0,1)
            p_perturb = self.random_perturb*self.perturb_decay**(t+1)

            # accept if good score or if passes threshold
            if  sim_score > curr_score or p < p_perturb:
                centroids, curr_score, feats = new_centroids, sim_score, newfeats

            # log images for debugging
            if self.save_plots and t % 20 == 0:
                mask = self.make_mask(centroids)
                im=self.make_pattern(mask,centroids, preds)
                plt.imshow(im)
                plt.colorbar()
                pth = os.path.join(self.save_path, f"annealed_pattern_step_{t}.jpeg")
                plt.savefig(pth)
                plt.close("all")

        mask = self.make_mask(centroids)
        im = self.make_pattern(mask, centroids, preds)
        if self.save_plots:
            plt.imshow(im)
            plt.colorbar()
            pth = os.path.join(self.save_path, "annealed_pattern.jpeg")
            plt.savefig(pth)
            plt.close("all")

        return centroids

    def candidate(self, centers, ix):
        test_centers = centers.copy()
        
        nx, ny, ix = self.random_move(test_centers, ix) #im_size,

        test_centers[ix] = (nx, ny)
        sim_score, preds, newfeats = self.evaluate(test_centers)
        
        return test_centers, sim_score, preds, newfeats

    def random_move(self, centers, ix): #im_size,
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
            
            found_valid_move = self.validate(newx, newy, nbors)
        return newx, newy, random_index

    def validate(self, cx, cy, centroids):
        # check if out of bounds   
        cxbool = self.out_of_bounds_check(cx)
        cybool = self.out_of_bounds_check(cy)
        
        # check centroid to centroid distance
        rep = np.tile(np.array([cx, cy]).reshape(-1,2), [len(centroids), 1])
        centroids_arr = np.array(centroids)
        dist = np.sqrt(np.sum((rep-centroids_arr)**2, axis = 1))
        valid = np.all(dist > self.c_to_c_dist)
        
        check = cxbool and cybool and valid

        return check

    def out_of_bounds_check(self, coord):
        if (coord < self.min_dist) or (coord > self.mask_size - self.min_dist):
            return False
        else:
            return True

    def evaluate(self, centroids):
        #new_feats = []
        mask = np.zeros((self.mask_size, self.mask_size))
        feats = self.extract_features(mask, centroids, messages=False)
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
        
    def sample_pattern(self, ):
        print("Sampling...")
        # create empty pattern
        mask = np.zeros((self.mask_size, self.mask_size))
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
                score, _, _ = self.evaluate(test_cents)
                sim_organoids.append((score, x, y))
            
            # select best simulated organoid
            best_sim_organoid = sorted(sim_organoids)[-1]
            newx, newy = best_sim_organoid[1], best_sim_organoid[2]
            mask = self.circle(newx, newy, mask, self.org_rad, fill = True)
            centroids.append((newx, newy))

        centroids = np.array(centroids)
        gen_centroids = self.shift(centroids)

        if self.save_plots:
            mask = self.make_mask(gen_centroids)
            gen_pattern = self.make_pattern(mask, gen_centroids)
            plt.imshow(gen_pattern)
            pth = os.path.join(self.save_path, "sampled_pattern.jpeg")
            plt.savefig(pth)
            plt.close("all")

        return gen_centroids

    def random_location(self, image):
        found_valid = False
        
        while not found_valid:
            x, y = np.random.choice(self.mask_size, 2)
            # ensure organoid is at least 2 pads worth away from wall and organoids
            if (x > self.min_dist and x < self.mask_size-self.min_dist and 
                y > self.min_dist and y < self.mask_size-self.min_dist):
                bool_mat = self.circle(x, y, image, self.min_dist)
                
                # check if overlapping with other organoids
                if np.sum(image[bool_mat]) == 0: 
                    found_valid = True
        return x,y

    