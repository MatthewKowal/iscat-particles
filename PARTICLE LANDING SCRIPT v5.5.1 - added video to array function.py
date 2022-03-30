# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:06:10 2021

@author: mdkowal@chem.ubc.ca

Particle Landing Calculations for IScat (Interferometric scattering microscopy)

Purpose:
    The purpose of this script is to quantify nanoparticle landings on a coverslip.
    Quantification is performed by measuring the interference signal of a particle 
    with a coverslip. This signal reaches a maximum...
    
    
    
    
    
Version History:
    v1
    v2
    v3
    v4
    v5
        -Optimizing the particle identification to detect:
            Material: 50nm PS beads
            Coverslip: PDL
            Field of View: 12um x 12um
            ResolutionL 512 x 512 px
        
        -Changed parameters to find more circles initially, then made the circle
        retention process more strict.
            Loosened circularity threshold
            Broadened range of radii
            
        -Changed the method used for checking the current list of particles.
        This works but has caused the program to be ridiculously slow. I will
        fix this in the next version

        -Added coaxial circle detection and removal.
        This fixes the problem of overlapping circles and false positives

    v5.1
        -This is my attempt to speed up the process of checking particles
        against the database
            -When checking if the particle is already on the list, only check
            the last ~500 particles or so. It seems like I could do less, but
            I was cautious because I'm initially pretty loose with what I
            consider a "particle" so I wind up with a lot of particles at first
            This was the biggest improvement in speed.
            -Used just one radius for Hough Transform. This was the second
            biggest improvement in speed.

    v5.2
        -Particle Images are saved as 16-bit ratiometric images centered at 1.
        -For 50nm PS particles the contrast ranges from ~ 0.79 ~ 1.1
        -Fit a Gaussian function to the particle image data

    v5.3
        I though the Gaussian fit was just ok so I tried to implement a Zero
        Order Bessel Function of the First Kind, but I couldn't figure out how
        to parameterize the function over a surface like I can with a separable
        Gaussian Function. A Bessel Function can be approximated with a
        "Difference of Gaussians" or a "Laplacian of Gaussian". In practice, 
        Difference of Gaussians seems to better fit the data, which in turn
        means I can use it to reject False Positives.
        
        !!!!!!!!!!!      THIS IS AN IMPORTANT CHANGE     !!!!!!!!!!!!!
        !!!!!!!!!!!       ALL DATA SINCE MARCH 2022      !!!!!!!!!!!!!
        !!!!!!!!!!!         IS SAVED IN THIS NEW         !!!!!!!!!!!!!
        !!!!!!!!!!!        AND BETTER FORMAT             !!!!!!!!!!!!!
        
        !!!!!!!!!     DATA OLDER THAN MARCH 17th          !!!!!!!!!!!!
        !!!!!!!!!  SHOULD BE REPROCESSED WITH THIS CODE   !!!!!!!!!!!!

    v5.4
        Added random sampling 3d plots and contrast histogram
    
    v5.5 
        Changes made for streamlined code:
            -Reworked the ratiometric process to quewith collections.deque datatype
            instead of lists. it didnt seem to make it any faster as small frame numbers
            -Streamlined remove_blip_particle functions by removing call to a singly
            used method .isReal. removed .isReal
        
        Changes made for improved low SNR particle finding
            -Added gaussian blur as a preprocess step for particle finding
            -Removed goodness of fit requirement temporarily
        
        

"""


import os.path
import numpy as np
import cv2
import time
import tqdm

import pickle



#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import skimage.filters

from skimage.draw import circle_perimeter
#from skimage.util import img_as_ubyte






'''
##################################################################################
##############                Test Functions                    ##################
##################################################################################
'''
def gaus_blur(images, gsize, gsig):
    print("Performing Gaussian Blur...")
    n, x, y = images.shape
    gaus_vid_ = np.ndarray([n,x,y])
    for frame in range(n):
        gaus_vid_[frame] = cv2.GaussianBlur(images[frame], (gsize,gsize), sigmaX=gsig, sigmaY=gsig)
    gaus_vid = gaus_vid_.astype(np.double)
    return gaus_vid

def ratiometric_2frame(images):
    print("processing 2 frame ratiometric imaging...")
    n, x, y = images.shape
    ratio_vid_ = np.ndarray([n-1,x,y])
    for frame in range(n-1):
        ratio_vid_[frame] = cv2.divide(images[frame+1], images[frame], scale=128)
    ratio_vid = ratio_vid_.astype(np.double)
    return ratio_vid

def psuedoFF(images, gsize, gsig):
    print("Performing Psuedo Flat Fielding...")
    n, x, y = images.shape
    pFF_vid_ = np.ndarray([n,x,y])
    for frame in range(n):
        lowpass_frame = cv2.GaussianBlur(images[frame], (gsize,gsize), sigmaX=gsig, sigmaY=gsig)
        pFF_vid_[frame] = cv2.divide(images[frame], lowpass_frame, scale=128)
    pFF_vid = pFF_vid_.astype(np.double)
    return pFF_vid


def draw_frame_particles(image, frame_particle_list):
    #this is really just for testing purposes
    radii = 10
    x, y = image.shape #f, x, y = video_in.shape
    rgb_image = color.gray2rgb(image)    
    print("\tDrawing Particles on video")
    #draw each particle on the frames that it exists on
    for p in frame_particle_list: #go through list of particles    
        circy, circx = circle_perimeter(r = p.y_vec[0],
                                            c = p.x_vec[0],
                                            radius = radii,
                                            shape = [x,y])
        rgb_image[circy, circx] = (220, 20, 220)        
    return rgb_image



def generate_fake_particle_list(length):
    pimage             = np.zeros([50,50])
    fake_particle_list = [Particle(0,0,0,90,0, pimage)]*length
    
    
    for pID in range(length):
        #rr = random.randint(0, 9)
        #print(rr)
        fake_particle_list[pID].x_vec              = [1,2,3,4,5,6,7,8,9,10] 
        fake_particle_list[pID].y_vec              = [1,2,3,4,5,6,7,8,9,10] 
        fake_particle_list[pID].f_vec              = [1,2,3,4,5,6,7,8,9,10] 
        fake_particle_list[pID].average_dark_pixel = 90
        fake_particle_list[pID].pID                = pID
        
    return fake_particle_list




'''
##################################################################################
##################################################################################
##############                                                  ##################
##############                BINARY FILE MANAGEMENT            ##################
##############                                                  ##################
##################################################################################
##################################################################################
'''


def get_bin_metadata(binfile, printmeta=False):
    
    basepath, filename = os.path.split(binfile)
    name               = filename.split(".")[0]
    metadata           = name.split("_")
    
    date = metadata[0]
    time = metadata[1]
    fov  = int(metadata[3])
    x    = int(metadata[4])
    y    = int(metadata[4])
    fps  = int(metadata[5])
    
    filesize        = os.path.getsize(binfile)
    nframes         = int(filesize / (x * y))
    remaining_bytes = filesize%(x * y)
    

    if printmeta:         #print everything out
        print("\nFile Properties")
        print("\tLocation:\t\t\t", basepath)
        print("\tFilename:\t\t\t", filename)
        print("\tSquare FOV (um) :\t", fov)
        print("\tX Resolution : \t\t", x)
        print("\tY Resolution : \t\t", y)
        print("\tFrames per second:  ", fps)
        print("\tFile Size: \t\t\t", filesize)
        print("\tNumber of frames: \t", nframes)
        print("\tRunning time:(s) \t", (nframes/fps), " seconds")
        print("\tRunning time (m): \t", (nframes/fps/60), " minutes")
        print("\tRemaining Bytes: \t", remaining_bytes)

    return basepath, filename, name, nframes, fov, x, y, fps



def load_binfile_into_array(binfile, print_time=True): #open a binfile and import data into image array
    print("\n\nLoading binfile into memory...")
    if print_time: start=time.time()
    
    #get constants    
    basepath, filename, name, nframes, fov, x, y, fps = get_bin_metadata(binfile, printmeta=True) # get basic info about binfile
    
    # import video
    dt = np.uint8                                     # choose an output datatype
    images = np.zeros([nframes, x, y], dtype=dt)      # this will be the output var
    print("Importing....")
    for c in tqdm.tqdm(range(nframes)):
        #print(c)
        frame1d = np.fromfile(binfile, dtype=np.uint8, count=(x*y), offset=(c*x*y))                      # read one chunk (a frame) from binary file as a row vector
        frame2d = np.reshape(frame1d, (x,y))                                                             # reshape the row vector as a 2d fram (x,y)
        images[c] = np.array(frame2d, dtype=dt)                                                          # add the frame to the output array of images
        if np.min(images[c] < 0):   print("WARNING, FRAME ", c, " OUT OF RANGE: ", np.min(images[c]))    # warn if minimum < 0 
        if np.max(images[c] > 255): print("WARNING, FRAME ", c, " OUT OF RANGE: ", np.max(images[c]))    # warn if maximum > 255
    #save_vid(images, 24, video_file)
    #printout = "saved " + str(video_file) + " OK!\n\n"
    #print(type(video_file))
    #print(type(printout))
    #return printout
      
    if print_time:
        end=time.time()
        print("\n\n\t Binfile Size:\t\t ", os.path.getsize(binfile) / 1E6, "Megabytes")
        print("\t Elapsed time (s):\t ", (end-start), " seconds")
        print("\t Elapsed time (m):\t ", ((end-start)/60), " minutes")
        print("\t Speed (Mbps):\t\t ", ( os.path.getsize(binfile) / 1E6) / (end-start), "Mb / s" )
        
    return images










'''
##################################################################################
##################################################################################
##############                                                  ##################
##############           RAW VIDEO --> PROCESSED VIDEO          ##################
##############                                                  ##################
##################################################################################
##################################################################################
'''

'''
    INPUT: A video as numpy array or maybe a list.
            Also some ask for a few paramters.
    
    OUTPUT: A processed video as a numpy array ([frames, x, y])


'''



def ratiometric_particle_finder(images, bufsize, clipmin, clipmax, print_time=True):
    
    #clipmin = clip_range[0]
    #clipmax = clip_range[1]
    #clipmin = 0.95
    #clipmax = 1.05
    #particle_list = False
    
    
    video_particle_list = []                      # an empty list of particles
    pID = 1
    
    
    
    
    if print_time: start = time.time()
    
    n,x,y = images.shape
    #print(n, x, y)
    print("\n\nPerforming Ratiometric Image Process + Particle Finding on ", n, " Frames\n")
    #ratio_vid16 = np.ndarray([n-2*bufsize,x,y]).astype(np.float16)
    ratio_image16 = np.ndarray([x,y]).astype(np.float16)
    ratio_image8  = np.ndarray([x,y]).astype(np.uint8)
    ratio_vid8    = np.ndarray([n-2*bufsize,x,y]).astype(np.uint8)
    #print(type(ratio_vid_))
    #print(ratio_vid_.shape)

    #generate 2 queues
    queue1 = []
    queue2 = []

    #create reatiometric video, iterate each frame in video
    for framenum in tqdm.tqdm(range(n)):
        
        
        #bring in a new frame for processing. Divide the new frame by its sum to normalize for frame brightness
        new_frame = images[framenum]/np.sum(images[framenum]) #new frame as the current frame, normalized bu dividing by its sum
        
        #fill ratiometric buffer
        if framenum < bufsize:#fill the first buffer
            queue1.append(new_frame)
            
        elif framenum < 2*bufsize:#fill the second buffer
            queue2.append(new_frame)
            
        #once the buffer is filled, calculate ratiometric frame and count particles
        else: #begin calculating ratiometric frames
            f          = framenum-2*bufsize     #this is a new counter for the ratiometric video since we lose total frames during processing
            queue1sum_ = np.sum(queue1, axis=0)  #sum queue1
            queue2sum_ = np.sum(queue2, axis=0)  #sum queue2
                         
            #divide queue1 by queue2, save as new 16-bit frame
            ratio_image16 = queue2sum_/queue1sum_
            
            
            #scale the 16-bit ratiometric image to fit an 8-bit grayscale video
            #p.clip(ratio_vid_, 0, 255).astype(np.uint8)
            ratio_image8 = np.clip( ((ratio_image16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
            ratio_vid8[f] = ratio_image8
                        
            #perform queue bookkeeping
            queue1.pop(0)                 #remove oldest element
            moved_frame = queue2.pop(0)   #move queue2 oldest element to queue1 part 1
            queue1.append(moved_frame)    #move queue2 oldest element to queue1 part 2
            queue2.append(new_frame)  #addest newest frame to queue2


            #find particles in the new ratiometric frame
            frame_particles = find_particles_in_frame(ratio_image8, ratio_image16)
            
            #go through the frame particles and add to or update our video particle list
            for p in range(len(frame_particles)):
            
            
                frame_particles[p].f_vec[0] = f
                frame_particles[p].pID = pID
                #average_dark_pixel = drkpxl_out[p]
                #print("avg drk pxl ", average_dark_pixel)
                #fullimage = vid_in[f]
                #pimage = fullimage[ (py-25):(py+25), (px-25):(px+25) ]
                #new_particle = Particle(px, py, first_frame_seen, average_dark_pixel, pID, pimage)
                   
                
                #if the video particle_list is empty, add the first particle to it
                if not video_particle_list:
                    video_particle_list.append(frame_particles[p])
                    pID += 1
                    #print("Added First Particle to the list")
                    
                else: #if particle_list is not empty, check to see if a similar particle exists
                    #print ("\tChecking Particle ", p, " of ", len(new_particles), " in frame ", f)
                    #print(len(particle_list))
                    #matchID = frame_particles[p].check_against_particle_list(video_particle_list)
                    
                    matchID = look_for_matching_particle(frame_particles[p], video_particle_list)
                    
                    #print(matchID) # matchID will either be false or a number greater than zero
                    if matchID: #if the particle is on the list already then update its info
                        #print("matched particle ID: ", matchID)
                        video_particle_list[matchID-1].updateParticle(frame_particles[p])
                    #    #update_particle(pID, new_particle)
                    #    print("\t\tI should write code to update this particle")
                    
                    else: #if the particle is not on the list then add a new particle and updoot the pID counter
                        video_particle_list.append(frame_particles[p])
                        #print("Added another Particle to the list, pID: ", pID)
                        pID += 1
                    
    
    particle_list_out = np.array(video_particle_list)
    #print("\n\t", len(particle_list_out), "particles found")


            
    #all frames have been processed so we cast the ratiometric video to 8-bit
    #if print_time: print(((time.time()-start)/60), " minutes just to calculate ratiometric.")
    #print(" Converting back to 8-bit video. Please Wait...\n")
    #ratio_vid8 = np.clip(ratio_vid16, 0, 255).astype(np.uint8) 
    
    #for newframenum in ratio_vid:
    #    print(np.mean(newframenum), np.min(newframenum), np.max(newframenum))
    
    
    
    if print_time:
        end = time.time()
        print(" ")
        print("\t Resolution: \t\t\t\t\t", x, y, "px")
        print("\t Total Pixels per frame: \t\t", (x*y), "px")
        print("\t New number of frames: \t\t\t", n, "frames")
        print("\t Elapsed time: \t\t\t\t\t", (end-start), " seconds")
        print("\t Elapsed time: \t\t\t\t\t", ((end-start)/60), " minutes")
        print("\t Total Number of Particles: \t", pID, "particles")
        print("\t Speed (n-fps): \t\t\t\t", (n / (end-start)), " finished frames / sec" )

    return ratio_vid8, particle_list_out





'''
##################################################################################
##################################################################################
##############                                                  ##################
##############          PARTICLE LIBRARY MANAGEMENT             ##################
##############                                                  ##################
##################################################################################
##################################################################################
'''




def look_for_matching_particle(particle, particle_list):

    matched_pID            = False       # By default, the Particle is not found on the list
    spatial_tolerance      = 10          # pixels
    temporal_tolerance     = 2           # if a particle was found here within the last x number of frames, consider it to be the same particle
    plist_length_tolerance = 500         # how far back do you want to search for a match in the particle list?


                                                                                ####    IMPORTANT: ONLY LOOK THROUGH THE LAST 1000 OR
                                                                                ####      SO ENTRIES OTHERWISE IT WILL TAKE FOREVER
    if len(particle_list) < plist_length_tolerance:                             #if the particle list is still short, compare the particle to the whole thing
        for lpart in particle_list: # Compare youself to the given Particle Object list and see if you could already be on the list.
            if abs(particle.x_vec[0] - lpart.x_vec[-1]) < spatial_tolerance:
                if abs(particle.y_vec[0] - lpart.y_vec[-1]) < spatial_tolerance:
                    if abs(particle.f_vec[0] - lpart.f_vec[-1]) < temporal_tolerance:
                        #print("match found, pID", p.pID)
                        matched_pID = lpart.pID
    else:                                                                        # otherwise only compare youreself to the last 1000 particles
         for lpart in particle_list[-plist_length_tolerance:]:
            if abs(particle.x_vec[0] - lpart.x_vec[-1]) < spatial_tolerance:
                if abs(particle.y_vec[0] - lpart.y_vec[-1]) < spatial_tolerance:
                    if abs(particle.f_vec[0] - lpart.f_vec[-1]) < temporal_tolerance:
                        #print("match found, pID", p.pID)
                        matched_pID = lpart.pID           
                        
    #returns either False or the matched pID
    return matched_pID




class Particle:
    def __init__(self, x_pos, y_pos, first_frame_seen, average_dark_pixel, pID, pimage):
        self.x_vec            = [x_pos]
        self.y_vec            = [y_pos]
        self.f_vec            = [first_frame_seen]
        self.drkpxl_vec       = [average_dark_pixel]
        self.pID              = pID
        self.pimage_vec       = [pimage]
        self.peak_contrastG   = 1.0
        self.peak_contrastLoG = 1.0
        self.peak_contrastDoG = 1.0
        self.rmseG            = 1.0
        self.rmseLoG          = 1.0
        self.rmseDoG          = 1.0
        self.paramsG          = []
        self.paramsLoG        = []
        self.paramsDoG        = []
        
        
    def check_against_particle_list(self, particle_list):
        matched_pID = False    # By default, the Particle is not found on the list
        spatial_tolerance = 10 #pixels
        temporal_tolerance = 2 # if a particle was found here within the last x number of frames, consider it to be the same particle
        
        for p in particle_list: # Compare youself to the given Particle Object list and see if you could already be on the list.
            if abs(self.x_vec[0] - p.x_vec[-1]) < spatial_tolerance:
                if abs(self.y_vec[0] - p.y_vec[-1]) < spatial_tolerance:
                    if abs(self.f_vec[0] - p.f_vec[-1]) < temporal_tolerance:
                        #print("match found, pID", p.pID)
                        matched_pID = p.pID
                        
        #returns either False or the matched pID
        return matched_pID
        
    def updateParticle(self, new_particle): # Take in a new particle and use its specs to update yourself
        self.x_vec.append(new_particle.x_vec[0])
        self.y_vec.append(new_particle.y_vec[0])
        self.f_vec.append(new_particle.f_vec[0])
        self.drkpxl_vec.append(new_particle.drkpxl_vec[0])
        self.pimage_vec.append(new_particle.pimage_vec[0])

    # def isReal(self):
    #     # check to see how many frames they showed up for
    #     # check to see if 

    #     isreal = False
    #     #print(self.pID, len(self.f_vec), self.f_vec)
        
    #     #instead of checking for total frame vector length, check for continuous sections with length greater than 6
    #     #https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    #     #stepsize = 1
    #     #consections = np.split(self.f_vec, np.where(np.diff(self.f_vec) != stepsize)[0]+1)
    #     #print("\t\t consecutive sections: ", consections)
        
    #     if len(self.f_vec) > 6:
    #         #print("kept particle ", self.pID, "because it was here for at lest ", len(self.f_vec), " frames")
    #         isreal = True
    #         #print("\t", self.pID, len(self.f_vec), self.f_vec)
    #     return isreal
    
    def darkest_pixel(self):
        return np.min(self.drkpxl_vec)
    
    def darkest_frame(self):
        dp = np.argmin(self.drkpxl_vec)
        return self.f_vec[dp]
    
    def darkest_pimage(self):
        dp = np.argmin(self.drkpxl_vec)
        return self.pimage_vec[dp]
    
    def darkest_x(self):
        dp = np.argmin(self.drkpxl_vec)
        return self.x_vec[dp]

    def darkest_y(self):
        dp = np.argmin(self.drkpxl_vec)
        return self.y_vec[dp]





'''
##################################################################################
##################################################################################
##############                                                  ##################
##############                 DATA --> VIDEO FILES             ##################
##############                                                  ##################
##################################################################################
##################################################################################
'''

from PIL import Image, ImageDraw, ImageFont



def save_bw_video(images, framerate, basepath, filename):
    print("Saving Video...")
    
    filename = filename + ".mp4"
    save_file_path = os.path.join(basepath, "output", filename)
    
    #nimg=34
    #width, height = 1024, 1024
    n, x, y = images.shape
    print(n,x,y)
    #fourcc = cv2.VideoWriter_fourcc(*'MPG4V')
    fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
                        #use a different 4-byte code for other codecs
                        #https://www.fourcc.org/codecs.php
    video = cv2.VideoWriter(save_file_path, fourcc, framerate, (y, x), 0)
    for c in range(n):
        frame = images[c]
        video.write(frame)
    video.release()
    
    print("Video Saved as: \t", filename)






'''
##################################################################################
##################################################################################
##############                                                  ##################
##############                   Particle Data                  ##################
##############               Display & Spreadsheet              ##################
##############                                                  ##################
##################################################################################
##################################################################################
'''



# SAVE PARTICLE DATABASE AS A PICKLE FILE FOR LATER USE

def save_particle_data(particle_list, basepath, filename):
    # generate filepath and save figure as .png
    #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
    #filename = "OUTPUT - particle list.pkl"
    filename = filename + ".pkl"
    save_file_path = os.path.join(basepath, "output", filename)
    pickle.dump(particle_list, open(save_file_path, "wb"))
    return True

def load_particle_data(filepath):
    particle_list = pickle.load( open(filepath, "rb"))
    return particle_list

def print_particle_report(particle_list):
    print("\nParticle Report\n")
    #print total number of particles found
    print(" Total Number of Particles Found: ", len(particle_list))

    #print histogram of particle life times in unit of frames
    hol = np.zeros(30) #histogram of lifetimes
    for p in particle_list:
        lifetime = len(p.f_vec)
        hol[lifetime] += 1
    print("\n\n Histogram of Lifetimes: \n\t Frames \t particles")
    for c, v in enumerate(hol): print("\t  ", c, "\t", v)
    return True


def generate_landing_rate_csv(particle_list, nframes, fps, basepath, filename):
    
    print("Making .csv file for Landing Rate...")
    csv_filename = os.path.join(basepath, "output", (filename+"__Landing Rate__.csv"))
        
    # get the total numnber of frames, n
    #n = 0
    #for p in particle_list:
    #    if p.f_vec[-1] > n: n = p.f_vec[-1]
    #print("\n\t N: ", n)
    
    n = nframes
    #n = ratio_vid.shape[0]
    #print(n)
    
    
    # particles per frame, list
    ppf = np.zeros(n)
    c = 0                                                       # initialize a total particle counter
    for i, f in enumerate(ppf):                                 #loop through each frame of the video     
    
        pids = [p.pID for p in particle_list if p.f_vec[0] == i]   # returns a list of particle ID's (which are equivalent to particle counts)
                                                                # for all particles that first landed on this particular frame
        if pids: c = pids[-1]                                   # if a list of landed particles was created, then use the largest pID
                                                                # (the last particle on the list), as the new total particle count
        ppf[i] = c                                              # set the particles per frame to be the total number of particles found so far

    # seconds per frame list
    # this converts the x axis from frames to seconds
    spf = np.linspace(0, (n/fps), n)
    
    #print(spf, ppf)    
    
    np.savetxt(csv_filename, np.transpose([spf, ppf]), delimiter=',')
    print("\t Particles list saved as: ", csv_filename)




import pandas as pd
def generate_particle_list_csv(particle_list, basepath, filename):    
    
    #generate empty dataframes for .csv files
    df1 = pd.DataFrame() 
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    
    print("Making Particle Library Spreadsheets .csv file...") #create files
    csv_filename  = os.path.join(basepath, "output", (filename+"__Particle Library__.csv"))
    csv2_filename = os.path.join(basepath, "output", (filename+"__Particle Images X__.csv"))
    csv3_filename = os.path.join(basepath, "output", (filename+"__Particle Images Y__.csv"))
    csv4_filename = os.path.join(basepath, "output", (filename+"__Particle Images XY__.csv"))
    
    #generate empty lists to store 
    pIDs                = np.zeros(len(particle_list))
    lifetimes           = np.zeros(len(particle_list))
    average_dark_pixels = np.zeros(len(particle_list))
    midfs               = np.zeros(len(particle_list))
    #xs                  = np.zeros(len(particle_list))
    #ys                  = np.zeros(len(particle_list))
    
    rmsesG              = np.zeros(len(particle_list))
    rmsesLoG            = np.zeros(len(particle_list))
    rmsesDoG            = np.zeros(len(particle_list))
    peak_contrastsG     = np.zeros(len(particle_list))
    peak_contrastsLoG   = np.zeros(len(particle_list))
    peak_contrastsDoG   = np.zeros(len(particle_list))
    
    #pimages             = np.zeros([len(particle_list),50,50])
    frames              = [0]*(len(particle_list))
    hlines              = [0]*(len(particle_list))
    vlines              = [0]*(len(particle_list))
    dots                = np.zeros(len(particle_list))


    # iterate through the particle list
    for c, v in enumerate(particle_list):
             
        #assign particle data to fill in a row of the .csv
        midf                   = int(np.around(len(v.f_vec)/2))
        midfs[c]               = midf
        pIDs[c]                = v.pID
        lifetimes[c]           = len(v.f_vec)
        frames[c]              = v.f_vec
        average_dark_pixels[c] = v.drkpxl_vec[midf]
        
        rmsesG[c]               = v.rmseG
        rmsesLoG[c]             = v.rmseLoG
        rmsesDoG[c]             = v.rmseDoG
        peak_contrastsG[c]      = v.peak_contrastG
        peak_contrastsLoG[c]    = v.peak_contrastLoG
        peak_contrastsDoG[c]    = v.peak_contrastDoG
        #xs[c] = len(v.pimage_vec)
        #ys[c] = len(v.pimage_vec)
        
        #print(v.pID, v.x_vec[midf], v.y_vec[midf], v.pimage_vec[midf].shape)
        pimage = np.array(v.pimage_vec[midf])
        
        #print(pimage)
        hlines[c] = pimage[25]
        vlines[c] = pimage[:][25]
        #print(hlines[c])
        #print(len(hlines[c]))
        
        df2[str(v.pID)] = hlines[c]
        df3[str(v.pID)] = vlines[c]
        df4[str(v.pID)] = hlines[c]+vlines[c]
        dots[c]         = np.dot(  (hlines[c]/256),(vlines[c]/256)    )
        
        # print("\nParticl: ",  v.pID)
        # print("\tX Loc: ",    v.x_vec[midf])
        # print("\tY Loc: ",    v.y_vec[midf])
        # print("\tLifetime: ", len(v.f_vec))
        # print("\tFrames: ",   v.f_vec)
        # print("\tpimage: \n", v.pimage_vec[midf])
        # print("\tLength of pimage_vec", len(v.pimage_vec))
        # print("\tParticle mid frame:", midf)
    
    df1["pID"]      = pIDs
    df1["lifetime"] = lifetimes
    # #df1["x"] = xs
    # #df1["y"] = ys
    df1["mid f"]              = midfs
    df1["average dark pixel"] = average_dark_pixels 
    df1["frames"]             = frames
    df1["x y dot product"]    = dots
    
    df1["RMSE G"]   = rmsesG
    df1["RMSE LoG"] = rmsesLoG
    df1["RMSE DoG"] = rmsesDoG
    df1["Peak Contrast G"]   = peak_contrastsG
    df1["Peak Contrast LoG"] = peak_contrastsLoG
    df1["Peak Contrast DoG"] = peak_contrastsDoG
    
    
    df1.to_csv(csv_filename)
    print("\t Particles list saved as: ", csv_filename)
    
    df2.to_csv(csv2_filename)
    df3.to_csv(csv3_filename)
    df4.to_csv(csv4_filename)
    



'''
##############################################################################
##############################################################################
########                                                         #############
########                                                         #############
########             GAUSSIAN FITTING FUNCTIONS                  #############
########                                                         #############
########                                                         #############
##############################################################################
##############################################################################
'''

from scipy import optimize
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error
from math import sqrt
import math


def gaussian(height, center_x, center_y, width_x, width_y, z_offset): #gaussian lamda function generator
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: z_offset + height*np.exp(-(((center_x-x)/width_x)**2 + ((center_y-y)/width_y)**2)/2)


def LoG(height, center_x, center_y, sigma):
    ox = lambda x: center_x - x
    oy = lambda y: center_y - y
    return lambda x, y: -(height*1800)/(math.pi*sigma**4)*(1-((ox(x)**2+oy(y)**2)/(2*sigma**2)))*np.exp(-((ox(x)**2+oy(y)**2)/(2*sigma**2)))+1


def DoG(height1, height2, center_x, center_y, sigma1, sigma2):
    #throw away height2, using it causes the optimize function to run past the max number of iterations its willing to. using the same height for each gaussian seems to work well anyway
    ox = lambda x: center_x - x
    oy = lambda y: center_y - y
    return lambda x, y: -2*height1*np.exp(-((ox(x)/sigma1)**2 + ((oy(y))/sigma1)**2)/2) + height1*np.exp(-((ox(x)/sigma2)**2 + ((oy(y))/sigma2)**2)/2) + 1


def fitgaussian(data): #find optimized gaussian fit for a particle
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    #first guess parameters
    height   = -0.1
    x        = 25.0
    y        = 25.0
    width_x  = 5.0
    width_y  = 5.0
    z_offset = 1.0
    params = height, x, y, width_x, width_y, z_offset
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def fitLoGaussian(data): #find optimized gaussian fit for a particle
    #first guess parameters
    height   = 0.7
    center_x = 25.0
    center_y = 25.0
    sigma    = 6.0
    params = height, center_x, center_y, sigma
    errorfunction = lambda p: np.ravel(LoG(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def fitDoGaussian(data): #find optimized gaussian fit for a particle
    #first guess parameters
    height1   = 0.25#0.3
    height2   = 0.2#0.4
    center_x = 25.0
    center_y = 25.0
    sigma1   = 6.0#4.2
    sigma2   = 10.0#5.7
    
    params = height1, height2, center_x, center_y, sigma1, sigma2
    errorfunction = lambda p: np.ravel(DoG(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p






'''
##################################################################################
##################################################################################
##############                                                  ##################
##############                   DATA --> IMAGES                ##################
##############                                                  ##################
##################################################################################
##################################################################################
'''


def draw_particle_landing_map(particle_list, mpp, sample_name, basepath, name):

    #n = len(particle_list)
    #x, y = np.zeros(n), np.zeros(n)
    #allx, ally = [], [] #get data into the right shape
    #for c in range(len(particle_list)):
    #    allx += particle_list[c].x_vec
    #    ally += particle_list[c].y_vec
    x = [p.x_vec[0]*mpp for p in particle_list]
    y = [p.y_vec[0]*mpp for p in particle_list]
    
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.figure(dpi=150)
    plt.scatter(x, y, s=1) # plot data
    plt.xlabel('microns')
    plt.ylabel('microns')
    title = "Particle Landing Locations, " + sample_name
    plt.title(title)
    
    # generate filepath and save figure as .png
    #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
    #filename = "OUTPUT - particle map.png"
    filename = "Particle Mqp " + name + ".png"
    save_file_path = os.path.join(basepath, "output", filename)
    plt.savefig(save_file_path)
    #plt.show()



def draw_one_particle_landing_v(p, sample_name, basepath, name):     # p is a Particle Object
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.figure(dpi=150)
    plt.scatter(p.f_vec, p.drkpxl_vec)
    plt.xlabel('frame number')
    plt.ylabel('Dark Pixel Value')
    title = "Scattering intensity, " + sample_name + ", Particle # " + str(p.pID)
    plt.title(title)
    
    
    #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
    #filename = "OUTPUT - particle " + str(p.pID) +  " landing.png"
    filename = "Landing V " + name + " pID", str(p.pID) + ".png"
    save_file_path = os.path.join(basepath,"output", filename)
    plt.savefig(save_file_path)
    




def save_3d_contour(data, fit, rmse, pID, functype, basepath, name): #generate a plot for the particle and its gaussian fit
    
    #print("Making Particle Library Spreadsheets .csv file...") #create files
    if not os.path.exists(os.path.join(basepath, "output", "random sampling")): os.makedirs(os.path.join(basepath, "output", "random sampling"))
    image_filename  = os.path.join(basepath, "output", "random sampling", (name+"-pID"+str(pID)+functype+".png"))
    
    x_dim, y_dim, x_steps, y_steps = 50, 50, 50, 50
    fig = plt.figure()

    x, y = np.mgrid[0:50, 0:50] #numpy.mgrid[-x_dim/2:x_dim/2:x_steps*1j, -y_dim/2:y_dim/2:y_steps*1j]
    v_min = np.min(fit)#0.7 #numpy.min(0)
    v_max = 1.05#np.max(data)#1.3 #numpy.max(255)

    ax = fig.gca(projection='3d')

    ax.contourf(x, y, data, zdir='z', levels=256, offset=v_min, cmap=cm.gray)
    
    #cset = ax.contourf(x, y, data, zdir='x', offset=-x_dim/2-1, cmap=cm.coolwarm)
    #cset = ax.contourf(x, y, data, zdir='y', offset=0, cmap=cm.coolwarm)

    ax.plot_wireframe(x, y, fit, rstride=5, cstride=5, alpha=0.5, color='blue', linewidth=1)
    ax.plot_surface(x, y, fit, rstride=2, cstride=2, alpha=0.3, cmap=cm.jet, linewidth=1)

    #ax.plot_wireframe(x, y, data, rstride=5, cstride=5, alpha=0.2, color='blue', linewidth=1)
    #ax.plot_surface(x, y, data, rstride=2, cstride=2, alpha=0.1, cmap=cm.jet, linewidth=1)


    ax.set_xlabel('X')
    #ax.set_xlim(-x_dim/2-1, x_dim/2+1)
    ax.set_ylabel('Y')
    #ax.set_ylim(-y_dim/2-1, y_dim/2+1)
    ax.set_zlabel('Z')
    #ax.set_zlim(v_min, v_max)
    
    ax.set_xlim([0,50])
    ax.set_ylim([0,50])
    ax.set_zlim([v_min,v_max])
    elev = 18
    azim = 127
    plt.gca().view_init(elev, azim)

    text_kwargs = dict(ha='left', va='center', fontsize=12, color='black')
    fig.text(0.18, 0.21, ("pID: " + str(pID)), **text_kwargs)
    fig.text(0.18, 0.18, ("RMSE: " + str(rmse)[:7]), **text_kwargs)
    fig.text(0.18, 0.15, ("F(x, y): " + functype), **text_kwargs)
    
    plt.savefig(image_filename)
    #plt.close()
    plt.show()



#import random
def random_sampling(particle_list, nsamples, basepath, name):
    
    if nsamples >= len(particle_list): sampleIDs = np.linspace(0,(len(particle_list)-1), num=len(particle_list)).astype('int')
    else: sampleIDs = np.random.randint(0,len(particle_list), size=nsamples)
    
    print(len(sampleIDs))
    print(sampleIDs)
    
    for i in sampleIDs:
        
        p = particle_list[i]
        
        darkest_frame = np.argmin(p.drkpxl_vec)
        darkest_image = p.pimage_vec[darkest_frame]
        
        #generate a space the same shape as the image to use for the fit
        Xin, Yin = np.mgrid[0:50, 0:50]         #emtpy grid to fit the parameters to. must be the same size as the particle iamge
        
        #generate an optimized parameters for gaussian fit
        paramsG          = fitgaussian(darkest_image)
        paramsLoG        = fitLoGaussian(darkest_image)
        paramsDoG        = fitDoGaussian(darkest_image)
        
        # Plot the fit function on a surface 
        fitG             = gaussian(*paramsG)(Xin, Yin)        
        fitLoG           = LoG(*paramsLoG)(Xin, Yin)        
        fitDoG           = DoG(*paramsDoG)(Xin, Yin)        
        
        # Calculate the PEAK CONTRAST based on the fit
        peak_contrastG   = 1 + paramsG[0]
        peak_contrastLoG = np.min(fitLoG)#1 + params[0]
        peak_contrastDoG = np.min(fitDoG)#1 + params[0]
        
        #calculate RMSE for the fit
        rmseG   = sqrt(mean_squared_error(darkest_image, fitG))
        rmseLoG = sqrt(mean_squared_error(darkest_image, fitLoG))
        rmseDoG = sqrt(mean_squared_error(darkest_image, fitDoG))


        save_3d_contour(darkest_image, fitG, rmseG, p.pID, "Gaussian", basepath, name)
        save_3d_contour(darkest_image, fitLoG, rmseLoG, p.pID, "Laplacian of Gaussian", basepath, name)
        save_3d_contour(darkest_image, fitDoG, rmseDoG, p.pID, "Difference of Gaussian", basepath, name)
        print("RMSE Gaussian: ", rmseG)
        print("RMSE LoG:      ", rmseLoG)
        print("RMSE DoG:      ", rmseDoG)
    

def plot_contrast_histogram(particle_list, basepath, name):
    if not os.path.exists(os.path.join(basepath, "output")): os.makedirs(os.path.join(basepath, "output"))
    image_filename  = os.path.join(basepath, "output", ("Contrast Histogram " + name + ".png"))
    
    contrasts = np.ones_like(particle_list)
    for i, p in enumerate(particle_list):
        #print(p.peak_contrastDoG)
        contrasts[i] = p.peak_contrastDoG 
    plt.hist(contrasts, bins=50)
    plt.gca().set(title='Contrast Histogram', ylabel='counts', xlabel='contrast')
    plt.xlim(0.5, 1)
    plt.savefig(image_filename)
    #plt.show()
        
#particle_list3 = remove_non_gaussian_particles(particle_list2)
#plot_contrast_histogram(particle_list3, basepath, name)



''' 
      ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO     
    ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
    ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
  ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
  ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
  ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
  ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
    ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO  
    ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
      ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO ###### TESTING SCENARIO 
'''

def save_particle_video(video_in, particle_list, framerate, basepath, name, extra_name):
    
    rgb_video = draw_particles_on_video(video_in, particle_list)
    #skvideo.io.vwrite(filename, images)
    print("Saving Video...")
    #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
    #filename = "OUTPUT - color video.avi"
    filename = name + "-color" + extra_name + ".avi"
    save_file_path = os.path.join(basepath, "output", filename)
    print("\t", save_file_path)
    # basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings\output"
    # filename = "OUTPUT - particle map.png"
    # save_file_path = os.path.join(basepath, "output", filename)
    #print("\t Saving Video: \t", filename)
    #nimg=34
    #width, height = 1024, 1024
    n, x, y, colors = rgb_video.shape
    #print("\t ", n,x,y, colors)
    video_out = np.ones_like(rgb_video)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = 0x00000021 #this is the 4-byte code (fourcc) code for mp4 codec
                        #use a different 4-byte code for other codecs
                        #https://www.fourcc.org/codecs.php
    videoObject = cv2.VideoWriter(save_file_path, fourcc, framerate, (y, x), isColor = True)
    #print(rgb_video_in.shape, "     ", video_out.shape)
    for c in range(n):
        #video_out[c] = rgb_video_in[c]
        videoObject.write(rgb_video[c])
    videoObject.release()
    
    #plt.imshow(images[1])
    #plt.show()
    
    return #video_out

def draw_particles_on_video(video_in, particle_list):
    radii = 10
    f, x, y = video_in.shape
    #rgb_image = cv2.cvtColor(video_in[0],cv2.COLOR_GRAY2RGB)
    video_RGB = []
    #video_RGB = np.ones([f, x, y, 3])
    print("\nDrawing particles on video...")
    print("\tFrame Dimensions: \t\t\t", x, " x ", y)
    #print("Video out shape:  ", video_RGB.shape)
    print("\tNumber of frames: \t\t\t", f)
    print("\tNumber of particles to Draw: \t", len(particle_list))
    
    
    
    #first, convert video to RGB
    print("Converting grayscale video to RGB...")
    for c, f in enumerate(video_in):
        #print("Frame: ", f, "Input Frame Shape: ", video_in[f].shape, " Output Frame Shape: ", video_RGB[f].shape)
        #
        rgb_image = color.gray2rgb(video_in[c])
        video_RGB.append(rgb_image)#[c] = color.gray2rgb(f)   
        #video_RGB[c] = cv2.cvtColor(f,cv2.COLOR_GRAY2BGR)
        #video_RGB[c] = cv2.merge([f, f, f])    
    video_out = np.array(video_RGB)
    #print("RGB Video Shape: ", video_out.shape)
    
    print("Drawing Framenumbers on video...")
    for f in range(len(video_out)):
        pillowImage = Image.fromarray(video_out[f])
        draw = ImageDraw.Draw(pillowImage)
        font = ImageFont.truetype("arial.ttf", 32)
        draw.text( (2,2), str(f), (10,10,10), font=font)
        
        video_out[f] = np.array(pillowImage, np.uint8)
    
    
    print("Drawing Particles on video...")
    #draw each particle on the frames that it exists on
    for c, p in enumerate(particle_list): #go through list of particles
        #print("Particle: ", p.pID, " found in frames: ", p.f_vec)
        
        for cc, f in enumerate(p.f_vec): #for this particle, go through the frame vector and draw the particle on the frame
            #print("\tDrawing particle ", p.pID, "on frame: ", f)
            image_copy = video_out[f]
            
            circy, circx = circle_perimeter(r = p.y_vec[cc],
                                            c = p.x_vec[cc],
                                            radius = radii,
                                            shape = [x,y])
            image_copy[circy, circx] = (220, 20, 220)
            
            pillowImage = Image.fromarray(image_copy)
            draw = ImageDraw.Draw(pillowImage)
            font = ImageFont.truetype("arial.ttf", 32)
            draw.text( (p.x_vec[cc],p.y_vec[cc]), str(p.pID), (220,20,220), font=font)
            
            image_copy = np.array(pillowImage, np.uint8)
            
            video_out[f] = image_copy
    print("Finished Drawing Particles on Video")
    return video_out



from collections import deque

def ratiometric_particle_finder2(images, bufsize, clipmin, clipmax):
    
    video_particle_list = []                      # an empty list of particles
    pID = 1
        
    start = time.time()
    
    n,x,y = images.shape
    #print(n, x, y)
    print("\n\nPerforming Ratiometric Image Process + Particle Finding on ", n, " Frames\n")
    #ratio_vid16 = np.ndarray([n-2*bufsize,x,y]).astype(np.float16)
    ratio_image16 = np.ndarray([x,y]).astype(np.float16)
    ratio_image8  = np.ndarray([x,y]).astype(np.uint8)
    ratio_vid8    = np.ndarray([n-2*bufsize,x,y]).astype(np.uint8)
    #print(type(ratio_vid_))
    #print(ratio_vid_.shape)
    

    #autofill the deques
    d1 = deque(images[:bufsize].astype(np.float16), bufsize)#[]
    d2 = deque(images[bufsize:(2*bufsize)].astype(np.float16), bufsize)#[]

    #print(len(d1))
    #print(len(d2))
    
    i16 = np.ndarray([x,y]).astype(np.float16)
    i8  = np.ndarray([x,y]).astype(np.uint8)
    v8  = np.ndarray([n-2*bufsize,x,y]).astype(np.uint8)

    #create reatiometric video, iterate each frame in video
    for f in tqdm.tqdm(range(n-2*bufsize)):
        
        # RATIOMETRIC
    
        #create ratiometric frame
        d1sum = np.sum(d1, axis=0).astype(np.float16)  #sum queue1
        d2sum = np.sum(d2, axis=0).astype(np.float16)
        i16 = d2sum/d1sum
        if f==0: info = [d1, d2, d1sum, d2sum, i16]
        
        #deque bookkeeping
        d1.append(d2.popleft())
        d2.append(images[f+2*bufsize])
        
        #save as 8-bit video
        i8 = np.clip( ((i16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
        v8[f] = i8
        
    
        # PARTICLE FINDING    
    
        #find particles in the new ratiometric frame
        frame_particles = find_particles_in_frame(i8, i16)
        #go through the frame particles and add to or update our video particle list
        for p in range(len(frame_particles)):
            frame_particles[p].f_vec[0] = f
            frame_particles[p].pID = pID
               
            #if the video particle_list is empty, add the first particle to it
            #else, look for a matching particle
            #   if a matching particle was found: update the video particle list
            #   else: create a new particle
            if not video_particle_list:
                video_particle_list.append(frame_particles[p])
                pID += 1
            else:
                matchID = look_for_matching_particle(frame_particles[p], video_particle_list)
                if matchID: video_particle_list[matchID-1].updateParticle(frame_particles[p])
                else: 
                    video_particle_list.append(frame_particles[p])
                    pID += 1
  
    
    particle_list_out = np.array(video_particle_list)
    #print("\n\t", len(particle_list_out), "particles found")

    end = time.time()
    print(" ")
    print("\t Resolution: \t\t\t\t\t", x, y, "px")
    print("\t Total Pixels per frame: \t\t", (x*y), "px")
    print("\t New number of frames: \t\t\t", n, "frames")
    print("\t Elapsed time: \t\t\t\t\t", (end-start), " seconds")
    print("\t Elapsed time: \t\t\t\t\t", ((end-start)/60), " minutes")
    print("\t Total Number of Particles: \t", pID, "particles")
    print("\t Speed (n-fps): \t\t\t\t", (n / (end-start)), " finished frames / sec" )

    return v8, particle_list_out, info




def find_particles_in_frame(image8, image16):  #find particles in a single frame
    '''
    Use this to find particles in a frame
    INPUT: A Greyscale Image as a numpy array (x, y), 8-bit clipped and 16-bit ratiometric centered at 1    
    OUTPUT: A List of Particle Objects for the frame'''


    mode = "50nm PS 500mW laser"
    
    if mode == "50nm PS 500mW laser":
        CANNY_SIGMA            = 1#3#4.5 #3   
        CANNY_LOW_THRESHOLD    = 50#11#10#9
        CANNY_HIGH_THRESHOLD   = 100#21#19    
        HOUGH_RADII = [8]#[8]  #8 works really well and its faster than radii ranges by around a factor of 2 or more usually
        # more hough radii comments
        #[4,5,6,7,8,9]#np.arange(6,9,1)#[8]#np.arange(6,18,3)#(6,21,1) worked well but its slow i think    8 #something from 8 -12 seems to work for 512x512 images #Can also use a range: np.arange(10, 11, 2) #default 25, 35, 2
        MIN_DIST= 10             # this is them minimum to distance required for two particles to be considered real (i.e. not overlapping)
        GAUSS_SIGMA            = 3.0    # KERNAL SIZE FOR INITIAL GAUSSIAN BLUR
        image8 = skimage.filters.gaussian(image8, sigma=GAUSS_SIGMA)*256 #sigma=3.0
        
        
    if mode == "50nm PS 5mW laser":
        CANNY_SIGMA            = 1#3#4.5 #3   
        CANNY_LOW_THRESHOLD    = 50#11#10#9
        CANNY_HIGH_THRESHOLD   = 100#21#19    
        HOUGH_RADII = [8]#[8]  #8 works really well and its faster than radii ranges by around a factor of 2 or more usually
        MIN_DIST= 10             # this is them minimum to distance required for two particles to be considered real (i.e. not overlapping)
        
        GAUSS_SIGMA            = 3.0    # KERNAL SIZE FOR INITIAL GAUSSIAN BLUR
        image8 = skimage.filters.gaussian(image8, sigma=GAUSS_SIGMA)*256 #sigma=3.0
    
        #binary threshold
        #BINARY_THRESHOLD       = 100    
        #image8 = image8 > BINARY_THRESHOLD#100
        

    frame_particle_list = []    # This is where we will store the output particle list for this frame
                                # it is a list of particle objects
    #detect edges
    
    edges = canny(image8, CANNY_SIGMA, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    #plt.imshow(edges)
    #plt.show()
    
    # Detect Circles from Edges
    #hough_radii = HOUGH_AIRY_DISK_RADIUS #[HOUGH_AIRY_DISK_RADIUS]
    hough_res = hough_circle(edges, HOUGH_RADII)
    #plt.imshow(hough_res[0]) #21 then 13
    #plt.show()
    
    #plt.imshow(hough_res[0]*255, cmap='Spectral')
    #plt.show()
    # Select the most prominent circles
    accums_, cx_, cy_, radii_ = hough_circle_peaks(hspaces = hough_res,             # Generate particle spec lists
                                               radii   = HOUGH_RADII,
                                               threshold = 0.3,                 # the lower this is the slower it goes but the more circles it finds. 0.3 is the best tradeoff 
                                               total_num_peaks = 30)
    
    #remove particles on edge
    cx, cy = [], []    
    for i in range(len(cx_)):
        if 25 < cx_[i] < (512-25) and 25 < cy_[i] < (512-25):
            cx.append(cx_[i])
            cy.append(cy_[i])
            #print(cx[-1])
    cx_, cy_ = cx, cy

        
    #check for co-axial circles
    cx, cy = [], []    
    for i in range(len(cx_)):
        #print("CHECKING PARTICLE:   ", i)
        if i == 0:     #add the first particle to the new list
            cx.append(cx_[0])
            cy.append(cy_[0])
        else:          #check to see if the current particle is already on the new list
            cx_[i] # current particle x
            cy_[i] # current particle_y
            
            cx[:]  #new list of known particles
            cy[:]  #new list of known particles
            
            match = 0
            for p in range(len(cx)): #if a particles x and y both are close to another particle, then mark it as matched
                if np.absolute(cx[p]-cx_[i]) < MIN_DIST and np.absolute(cy[p]-cy_[i]) < MIN_DIST: match += 1
            
          
            if match == 0:  #if it wasnt a match to any other particle then add it to the list
                cx.append(cx_[i])
                cy.append(cy_[i])
    
    #cx, cy = cx_, cy_        
            
        
    
    # Make a list of particles darker than some threshold
    # Also record the average value of the center  100 pixels
    # and record the particle image
    for c in range(len(cx)):
        px, py = cx[c], cy[c]
        avgdrkpxl = np.average(image16[ (py-5):(py+5), (px-5):(px+5) ])           # Calculate approximately how dark the particle is
        if avgdrkpxl < 1:                                                     # If its sufficiently dark, Create a new
            
        
            #it would be nice to fix this so it pastes the iamge on a grey background
            pimage = image16[ (py-25):(py+25), (px-25):(px+25) ]                  # Particle Object and add it to the output list
            
            
            frame_particle_list.append(Particle(cx[c], cy[c], 0, avgdrkpxl, 0, pimage))
    
    
    # Make a list of all particles
    # for c in range(len(cx)):
    #     px, py = cx[c], cy[c]
    #     avgdrkpxl = np.average(image[ (py-5):(py+5), (px-5):(px+5) ])           # Calculate approximately how dark the particle is
    #     pimage = image[ (py-25):(py+25), (px-25):(px+25) ]                  # Particle Object and add it to the output list
    #     frame_particle_list.append(Particle(cx[c], cy[c], 0, avgdrkpxl, 0, pimage))
    
    
    
    return frame_particle_list




def draw_langmuir(particle_list, nframes, fps, title, basepath, name):

    # particles per frame, list
    ppf = np.zeros(nframes)
    c = 0

    #for i, f in enumerate(ppf):                                 #loop through each frame of the video        
    for i in range(len(ppf)):
        pids = [p.pID for p in particle_list if p.f_vec[0] == i]   # returns a list of particle ID's (which are equivalent to particle counts)
                                                                # for all particles that first landed on this particular frame
        #print(pids)
        if pids: c = pids[-1]                                   # if a list of landed particles was created, then use the largest pID
                                                                # (the last particle on the list), as the new total particle count
        ppf[i] = c                                              # set the particles per frame to be the total number of particles found so far


    # seconds per frame list
    # this converts the x axis from frames to seconds
    spf = np.linspace(0, (nframes/fps), nframes)
    
    #print(spf, ppf)
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.figure(dpi=150)        
    plt.scatter(spf, ppf, s=1)
    plt.xlabel('Time /s')
    plt.ylabel('Number of Landings')
    plt.title(title)

    
    # generate filepath and save figure as .png
    #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
    #filename = "OUTPUT - Langmuir Landing Rate.png"
    filename = "Langmuir Landing Rate " + name + ".png"
    save_file_path = os.path.join(basepath, "output", filename)
    plt.savefig(save_file_path)
    #plt.show()


def remove_blip_particles(particle_list):
    st = time.time()
    print("\nRemoving Blip Particles...")
    
    cleaned_particle_list = []
    c = 1
    for p in particle_list:
        #print("length of f_vec: ", len(p.f_vec))
        if len(p.f_vec) > 6:
            cleaned_particle_list.append(p)
            cleaned_particle_list[-1].pID = c
            c += 1
            
    particle_list_out = np.array(cleaned_particle_list)
    
    et = time.time()
    #print("\tFinished....")
    print("\tPrticles before clean up:\t\t", len(particle_list))
    print("\tParticles left after clean up:\t", len(particle_list_out))
    print("\tElapsed time:\t\t\t\t\t", ((et-st)), " seconds")
    if (et-st) > 0: print("\tscanning speed:\t\t\t", (len(particle_list)/(et-st)), " Particles/sec")
    return particle_list_out




def remove_non_gaussian_particles(particle_list):
    
    particle_list_out = []
    new_pID = 1
    
    print("\nFitting: ", len(particle_list), " Particles to approximations of Zero Order Bessel Functions of the First Kind...")
    #print("pID  \t   drkpxl \t loc \t   len \t %  \t   RMSE  \t  peak contrast")
    for p in tqdm.tqdm(particle_list):  
        darkest_frame = np.argmin(p.drkpxl_vec)
        darkest_image = p.pimage_vec[darkest_frame]
        
        #generate a space the same shape as the image to use for the fit
        Xin, Yin = np.mgrid[0:50, 0:50]         #emtpy grid to fit the parameters to. must be the same size as the particle iamge
        
        #generate an optimized parameters for gaussian fit
        paramsG          = fitgaussian(darkest_image)
        paramsLoG        = fitLoGaussian(darkest_image)
        paramsDoG        = fitDoGaussian(darkest_image)
        
        # Plot the fit function on a surface 
        fitG             = gaussian(*paramsG)(Xin, Yin)        
        fitLoG           = LoG(*paramsLoG)(Xin, Yin)        
        fitDoG           = DoG(*paramsDoG)(Xin, Yin)        
        
        # Calculate the PEAK CONTRAST based on the fit
        peak_contrastG   = 1 + paramsG[0]
        peak_contrastLoG = np.min(fitLoG)#1 + params[0]
        peak_contrastDoG = np.min(fitDoG)#1 + params[0]
        
        #calculate RMSE for the fit
        rmseG   = sqrt(mean_squared_error(darkest_image, fitG))
        rmseLoG = sqrt(mean_squared_error(darkest_image, fitLoG))
        rmseDoG = sqrt(mean_squared_error(darkest_image, fitDoG))

        #update current particle
        p.pID              = new_pID
        p.rmseG            = rmseG
        p.rmseLoG          = rmseLoG
        p.rmseDoG          = rmseDoG
        p.peak_contrastG   = peak_contrastG
        p.peak_contrastLoG = peak_contrastLoG
        p.peak_contrastDoG = peak_contrastDoG
        p.paramsG          = paramsG
        p.paramsLoG        = paramsLoG
        p.paramsDoG        = paramsDoG
        
        
        ''' Remove Particles that do not fit the model well '''
        if rmseDoG < 0.1:
            #populate new particle list   
            particle_list_out.append(p)
            new_pID += 1
 
    particle_list_out = np.asarray(particle_list_out)
    
    return particle_list_out




# ''' INITIAL CONDITIONS   '''



binfile = r'C:/Users/user1/Desktop/python projects.2022/python iscat/video compression test/2022-03-22_16-30-52_raw_12_512_70_500mw.bin'

sample_name = "test test test"

output_framerate = 24  # frane rate for output video
binsize          = 10  # ratiometric bin size

clipmin = 0.95
clipmax = 1.05




''''IMPORT AND PROCESS BINARY FILE'''
binimages = load_binfile_into_array(binfile)

basepath, filename, name, nframes, fov, x, y, fps = get_bin_metadata(binfile) # get basic info about binfile
if not os.path.exists(os.path.join(basepath, "output")):
    os.makedirs(os.path.join(basepath, "output"))
# ########################################################
# # # MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES
start_frame  = 1000
end_frame    = 2000
images = binimages[start_frame:end_frame]
nframes = end_frame - start_frame
# # # OR USE THE FULL FILE
#images = binimages
# ########################################################
#save the original raw video

save_bw_video(images, output_framerate, basepath, name)

#%%


ratio_vid8, particle_list, info = ratiometric_particle_finder2(images, binsize, clipmin, clipmax)

save_bw_video(ratio_vid8, output_framerate, basepath, (name+"ratio"))
save_particle_video(ratio_vid8, particle_list, output_framerate, basepath, name, "1")


particle_list2 = remove_blip_particles(particle_list)
save_particle_video(ratio_vid8, particle_list2, output_framerate, basepath, name, "2")

particle_list3 = remove_non_gaussian_particles(particle_list2)
save_particle_video(ratio_vid8, particle_list3, output_framerate, basepath, name, "3")

#%%
''' Save Particle Data '''
save_particle_data(particle_list3, basepath, name)
# Generate data for Langmuir Adsorption rate
generate_landing_rate_csv(particle_list3, nframes, fps, basepath, name)
# Generate data for each particle
generate_particle_list_csv(particle_list3, basepath, name)

''' draw plots '''
draw_particle_landing_map(particle_list3, (fov / x), sample_name, basepath, name)      
#draw_one_particle_landing_v(particle_list2[47], sample_name, basepath, filename)
draw_langmuir(particle_list3, nframes, fps, sample_name, basepath, name)
random_sampling(particle_list3, 2, basepath, name)    
plot_contrast_histogram(particle_list3, basepath, name)




#%%


'''load from video file '''



import skvideo.io  
def load_raw_video(filepath):
    #d1 = skvideo.io.vread(filepath, as_grey=False) 
    d2 = skvideo.io.vread(filepath, as_grey=True) 
    #d3 = np.dot(d1[:][:][:],[0.3, 0.3, 0.3]).astype(np.uint8)
    d4 = np.dot(d2,[1]).astype(np.uint8)
    return d4

vidpath = r'C:/Users/user1/Desktop/python projects.2022/python iscat/video compression test/output/2022-03-22_16-30-52_raw_12_512_70_500mw.mp4'

images_fv = load_raw_video(vidpath)
save_bw_video(images_fv, output_framerate, basepath, (name+"_fromvideo"))
#%%

ratio_vid8_fv, pl_fv, info = ratiometric_particle_finder2(images_fv, binsize, clipmin, clipmax)
save_bw_video(ratio_vid8_fv, output_framerate, basepath, (name+"ratio-fromvid"))
save_particle_video(ratio_vid8_fv, pl_fv, output_framerate, basepath, name, "1-fromvid")

#%%
pl_fv2 = remove_blip_particles(pl_fv)
save_particle_video(ratio_vid8_fv, pl_fv2, output_framerate, basepath, name, "2-fromvid")
pl_fv3 = remove_non_gaussian_particles(pl_fv2)
save_particle_video(ratio_vid8_fv, pl_fv3, output_framerate, basepath, name, "3-fromvid")



#%%

''' Save Particle Data '''

save_particle_data(pl_fv3, basepath, name)
# Generate data for Langmuir Adsorption rate
generate_landing_rate_csv(pl_fv3, nframes, fps, basepath, name)
# Generate data for each particle
generate_particle_list_csv(pl_fv3, basepath, name)


''' draw plots '''
draw_particle_landing_map(pl_fv3, (fov / x), sample_name, basepath, name)      
#draw_one_particle_landing_v(particle_list2[47], sample_name, basepath, filename)
draw_langmuir(pl_fv3, nframes, fps, sample_name, basepath, name)
random_sampling(pl_fv3, 2, basepath, name)    
plot_contrast_histogram(pl_fv3, basepath, name)









#%%




''' RATIOMETRIC PROCESS AND PARTICLE FINDING '''

ratio_vid8, particle_list, info = ratiometric_particle_finder2(images, binsize, clipmin, clipmax)

save_bw_video(ratio_vid8, output_framerate, basepath, (name+"ratio"))
save_particle_video(ratio_vid8, particle_list, output_framerate, basepath, name, "1")

#%%

# ''' TEST ONE FRAME FOR PARTICLE IDENTIFICATION '''
# ''' test conditions. cropping from frame 1000:1232
#       frame      problem
#       20         white rim detected
#       21         same      
# '''
# img1 = ratio_vid[16]
# plt.imshow(img1)
# plt.show()
# frame_particle_list = find_particles_in_frame(img1)
# pimg = draw_frame_particles(img1, frame_particle_list)
# plt.imshow(pimg)
# plt.show()
#plt.imsave(r"C:\Users\Matt\Desktop\testing data\2022-02-18-50nm 1000x PDL 500mWlaser pH5\VIDEOS\output\out.png", pimg)



''' CLEAN UP PARTICLE DATA '''
particle_list2 = remove_blip_particles(particle_list)
save_particle_video(ratio_vid8, particle_list2, output_framerate, basepath, name, "2")

particle_list3 = remove_non_gaussian_particles(particle_list2)
save_particle_video(ratio_vid8, particle_list3, output_framerate, basepath, name, "3")
#particle_list3 = particle_list2


''' Save Particle Data '''

save_particle_data(particle_list3, basepath, name)
# Generate data for Langmuir Adsorption rate
generate_landing_rate_csv(particle_list3, nframes, fps, basepath, name)
# Generate data for each particle
generate_particle_list_csv(particle_list3, basepath, name)


''' draw plots '''
draw_particle_landing_map(particle_list3, (fov / x), sample_name, basepath, name)      
#draw_one_particle_landing_v(particle_list2[47], sample_name, basepath, filename)
draw_langmuir(particle_list3, nframes, fps, sample_name, basepath, name)
random_sampling(particle_list3, 2, basepath, name)    
plot_contrast_histogram(particle_list3, basepath, name)

    






#%%








'''
# script #### script #### script #### script #### script #### script ##########
## script #### script #### script #### script #### script #### script #########
### script #### script #### script #### script #### script #### script ########
#### script #### script #### script #### script #### script #### script #######
##### script #### script #### script #### script #### script #### script ######
###### script #### script #### script #### script #### script #### script #####
####### script #### script #### script #### script #### script #### script ####
######## script #### script #### script #### script #### script #### script ###
######### script #### script #### script #### script #### script #### script ##
'''




''' PROCESS A SINGLE .BIN FILE '''
def generate_exp_data(binfile, sample_name):
    st = time.time()

    # use this to process a single experiment video.
    # you will need a binfile of the video and a sample name
    
    
    ''' INITIAL CONDITIONS   '''
    output_framerate = 24  # frane rate for output video
    binsize          = 10  # ratiometric bin size
    clipmin          = 0.95
    clipmax          = 1.05

    
        
    ''''IMPORT AND PROCESS BINARY FILE'''
    binimages = load_binfile_into_array(binfile)
    basepath, filename, name, nframes, fov, x, y, fps = get_bin_metadata(binfile) # get basic info about binfile
    if not os.path.exists(os.path.join(basepath, "output")):
        os.makedirs(os.path.join(basepath, "output"))
    # ########################################################
    # # # MAKE SHORTENED VERSION OF VIDEO FOR TESTING PURPOSES
    start_frame  = 1000
    end_frame    = 1100
    images = binimages[start_frame:end_frame]
    # # # OR USE THE FULL FILE
    #images = binimages
    # ########################################################
    #save the original raw video
    save_bw_video(images, output_framerate, basepath, name)

    
    ''' RATIOMETRIC PROCESS AND PARTICLE FINDING '''
    ratio_vid8, particle_list= ratiometric_particle_finder(images, binsize, clipmin, clipmax, print_time=True)
    save_bw_video(ratio_vid8, output_framerate, basepath, (name+"-ratio"))


    
    ''' CLEAN UP PARTICLE DATA '''
    particle_list2 = remove_blip_particles(particle_list)
    particle_list3 = remove_non_gaussian_particles(particle_list2)
    save_particle_data(particle_list3, basepath, name)
    ''' Generate .csv Files from Particles '''
    # Generate data for Langmuir Adsorption rate
    generate_landing_rate_csv(particle_list3, nframes, fps, basepath, name)
    # Generate data for each particle
    generate_particle_list_csv(particle_list3, basepath, name)
    
    
    
    ''' GENERATE CIRCLED PARTICLE VIDEOS '''
    #generate video of images with circles drawn on particles
    video_with_circles = draw_particles_on_video(ratio_vid8, particle_list3)
    save_color_video(video_with_circles, output_framerate, basepath, (name+"-color"))
    
    
    ''' draw plots '''
    draw_particle_landing_map(particle_list3, (fov / x), sample_name, basepath, name)      
    #draw_one_particle_landing_v(particle_list2[47], sample_name, basepath, filename)
    draw_langmuir(particle_list3, nframes, fps, sample_name, basepath, name)
    random_sampling(particle_list3, 2, basepath, name)    
    plot_contrast_histogram(particle_list3, basepath, name)

    
    et = time.time()
    print("\n", name)
    print("\n\nFINSIHED... Processed ", os.path.getsize(binfile), " bytes in ", ((et-st)/60), " minutes\n\n")
    

binfile = r'C:/Users/user1/Desktop/python iscat/video compression test/2022-03-07_15-57-22_raw_12_512_70.bin' 

sample_name = "testP"

generate_exp_data(binfile, sample_name)



#%%

''' PROCESS A WHOLE FOLDER OF FOLDERS OF .BINS '''
def batch_process_exp_data(root_dir):
    
    binfiles = []
    folders = []
    sample_names = []

    for f in os.listdir(root_dir):
        
        sample_name = f[11:]
        folder = os.path.join(root_dir, f, 'VIDEOS')
        for filename in os.listdir(folder):
            if filename.endswith('.bin'):
                binfile = os.path.join(folder, filename)
                
                binfiles.append(binfile)
                folders.append(folder)
                sample_names.append(sample_name)
                
                print(binfile, sample_name)
                generate_exp_data(binfile, sample_name)
                
    return binfiles, folders, sample_names



#root_dir = r'C:\Users\Matt\Desktop\experiments\DATA - 2021.12 - photo pressure'
#root_dir = r'D:\______2022.March.03 - polystyrene on PDL at different basic pHs'
root_dir = r'D:\______2022.March.07 - 50nm PS on PDL at different ACIDIC pHs'
             
             

strt = time.time()

a, b, c = batch_process_exp_data(root_dir)

endt = time.time()
print("Total Runtime: ", ((endt-strt)/60), " minutes")




























#%%  PLOT AND SAVE A LANDING PARTICLE PLOT

#load_particle_path = r"C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\2021-11-08-50nm PS NPs 1000x\VIDEOS\output\OUTPUT - particle list - 2021-11-08_18-11-16_raw_512_512_212.pkl"
#sample_name = "50 nm PS nanobeads"

#headtail     = os.path.split(load_particle_path)
#basepath     = r'C:\Users\Matt\Desktop\ISCAT EXPERIMENTS\2021-11-08-50nm PS NPs 1000x\VIDEOS' #headtail[0]
#filename     = headtail[1].split('.')[0]

#particle_list3 = load_particle_data(load_particle_path)

#p=particle_list3[205]
#draw_one_particle_landing_v(p, sample_name, basepath, filename)


#%% create a .csv file of landings vs time


    
# def make_csv_from_particle_list_pkl(pfile, fps):

#     particle_list = load_particle_data(pfile)
#     basepath = os.path.split(pfile)[0]
#     filename = os.path.split(pfile)[1]
#     filename = filename[:-4]
#     csv_filename = os.path.join(basepath, (filename+"___.csv"))
    
#     print(basepath)
#     print(filename)
#     print(csv_filename)

#     # get the total numnber of frames, n
#     n = 0
#     for p in particle_list:
#         if p.f_vec[-1] > n: n = p.f_vec[-1]
#     #print("\n\t N: ", n)
    
    

#     # particles per frame, list
#     ppf = np.zeros(n)
#     c = 0                                                       # initialize a total particle counter
#     for i, f in enumerate(ppf):                                 #loop through each frame of the video        
#         pids = [p.pID for p in particle_list if p.f_vec[0] == i]   # returns a list of particle ID's (which are equivalent to particle counts)
#                                                                 # for all particles that first landed on this particular frame
#         if pids: c = pids[-1]                                   # if a list of landed particles was created, then use the largest pID
#                                                                 # (the last particle on the list), as the new total particle count
#         ppf[i] = c                                              # set the particles per frame to be the total number of particles found so far



#     # seconds per frame list
#     # this converts the x axis from frames to seconds
#     spf = np.linspace(0, (n/fps), n)
    
#     print(spf, ppf)
    
    
#     np.savetxt(csv_filename, np.transpose([spf, ppf]), delimiter=',')
    
#     #plt.figure(dpi=150)        
#     #plt.scatter(spf, ppf)
#     #plt.xlabel('Time /s')
#     #plt.ylabel('Number of Landings')
#     #plt.title(title)

    
#     # generate filepath and save figure as .png
#     #basepath = r"C:\Users\Matt\Desktop\particle tracking python\Particle Tracking - Coverslip landings"
#     #filename = "OUTPUT - Langmuir Landing Rate.png"
    
#     #filename = "OUTPUT - Langmuir Landing Rate - " + filename + ".png"
#     #save_file_path = os.path.join(basepath, "output", filename)
#     #plt.savefig(save_file_path)
    
#     #plt.show()

# #pfile = r'C:\Users\Matt\Desktop\50nm Alumina dust standard additions to 50nm PS particles\2022-01-24-water + 50nm PS particles\VIDEOS\output\OUTPUT - particle list - 2022-01-24_15-14-31_raw_512_512_212.pkl'
# #pfile = r'C:\Users\Matt\Desktop\50nm Alumina dust standard additions to 50nm PS particles\2022-01-24-20ul 50nm filtered dust\VIDEOS\output\OUTPUT - particle list - 2022-01-24_15-39-39_raw_512_512_212.pkl'
# #pfile = r'C:\Users\Matt\Desktop\50nm Alumina dust standard additions to 50nm PS particles\2022-01-24-10ul 50nm filtered dust + 10ul 50nm ps\VIDEOS\output\OUTPUT - particle list - 2022-01-24_15-52-48_raw_512_512_212.pkl'
# #pfile = r'C:\Users\Matt\Desktop\50nm Alumina dust standard additions to 50nm PS particles\2022-01-24-5ul water 5ul dust 10ul 50nm ps\VIDEOS\output\OUTPUT - particle list - 2022-01-24_16-13-18_raw_512_512_212.pkl'
# #pfile = r'C:\Users\Matt\Desktop\50nm Alumina dust standard additions to 50nm PS particles\2022-01-24-5ul 50nm filtered dust + 10ul 50nm ps\VIDEOS\output\OUTPUT - particle list - 2022-01-24_16-04-20_raw_512_512_212.pkl'
# #pfile = r'C:\Users\Matt\Desktop\50nm Alumina dust standard additions to 50nm PS particles\2022-01-24-2ul dust 8ul water 10ul 50nm ps\VIDEOS\output\OUTPUT - particle list - 2022-01-24_16-22-20_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/experiments/DATA - 2021.11 - Polystyrene NP Size and concentration experiments/2021-11-09-50nm PS NPs 4000x/VIDEOS/output/OUTPUT - particle list - 2021-11-09_16-30-38_raw_512_512_212.pkl'

# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 1000x PDL 500mWlaser pH5/VIDEOS/output/OUTPUT - particle list - 2022-02-18_16-04-13_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 2000x PDL 500mWLaser pH5/VIDEOS/output/OUTPUT - particle list - 2022-02-18_16-30-36_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 4000x PDL 500mWLaser pH5/VIDEOS/output/OUTPUT - particle list - 2022-02-18_17-05-57_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 4000x PDL 500mWLaser pH5/VIDEOS/output/OUTPUT - particle list - 2022-02-18_17-13-09_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 4000x PDL 500mWLaser pH5/VIDEOS/output/OUTPUT - particle list - 2022-02-18_17-31-43_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 4000x PDL 500mWLaser pH5 test2/VIDEOS/output/OUTPUT - particle list - 2022-02-18_18-45-42_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 4000x PDL 500mWLaser pH5 test3/VIDEOS/output/OUTPUT - particle list - 2022-02-18_20-35-34_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 4000x PDL 500mWLaser pH5 test4/VIDEOS/output/OUTPUT - particle list - 2022-02-18_20-38-35_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 4000x PDL 500mWLaser pH5 test4/VIDEOS/output/OUTPUT - particle list - 2022-02-18_20-52-31_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 8000x PDL 500mWLaser pH5/VIDEOS/output/OUTPUT - particle list - 2022-02-18_19-46-52_raw_512_512_212.pkl'
# #pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 8000x PDL 500mWLaser pH5 test2/VIDEOS/output/OUTPUT - particle list - 2022-02-18_20-18-21_raw_512_512_212.pkl'
# pfile = r'C:/Users/Matt/Desktop/ISCAT EXPERIMENTS/______2022.February.18 - 50nm PS on PDL at different concentrations/2022-02-18-50nm 10000x PDL 500mWLaser pH5/VIDEOS/output/OUTPUT - particle list - 2022-02-18_20-04-01_raw_512_512_212.pkl'


# make_csv_from_particle_list_pkl(pfile, 212)



#%% load videos into arrays. this was supposed to be a means of compressing
#our raw video because the bin files are so huge. unfortunately it wasnt able
#to find particles well. it might help to tweak the particle finding method,
#but it kind of defeats the purpose


# import skvideo.io  
# def videofile2array(vid_filepath):
#     videodata = skvideo.io.vread(vid_filepath)  
#     images = np.average(videodata, axis=3).astype(np.uint8)
#     return images

# vid_filepath = r'C:/Users/user1/Desktop/video compression test/output/2022-03-07_15-57-22_raw_12_512_70.mp4'
# images2 = videofile2array(vid_filepath)
# save_bw_video(images2, output_framerate, basepath, (name+"-Raw2"))


# ratio_vid8, particle_list= ratiometric_particle_finder(images2, binsize, clipmin, clipmax, print_time=True)
# save_bw_video(ratio_vid8, output_framerate, basepath, (name+"-ratio"))
# ''' CLEAN UP PARTICLE DATA '''
# particle_list2 = remove_blip_particles(particle_list)
# particle_list3 = remove_non_gaussian_particles(particle_list2)
# save_particle_data(particle_list3, basepath, name)
# ''' Generate .csv Files from Particles '''
# # Generate data for Langmuir Adsorption rate
# generate_landing_rate_csv(particle_list3, nframes, fps, basepath, name)
# # Generate data for each particle
# generate_particle_list_csv(particle_list3, basepath, name)

# ''' GENERATE IMAGES AND VIDEOS FROM DATA '''
# #generate video of images with circles drawn on particles
# video_with_circles = draw_particles_on_video(ratio_vid8, particle_list3)
# save_color_video(video_with_circles, output_framerate, basepath, name)
