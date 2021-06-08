#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import cv2 
import copy

cube_dim = 512
cube_base = np.float32([
            		[0, 0],
            		[cube_dim-1, 0],
            		[cube_dim-1, cube_dim-1],
            		[0, cube_dim-1]])

def find_tag(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Reference: https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    # Bilateral Filter has a nice property of removing noise while still preserving the actual edges
    blur = cv2.bilateralFilter(gray, 15, 75, 75)
    edges = cv2.Canny(blur, 50, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    squares = []
    # ===========================================================================
    # Check the dimension of the square
    # for c in contours:
    #     x,y,w,h = cv2.boundingRect(c)
    #     cv2.putText(frame, str(w*h), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 1)
    # =============================================================================
    
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4:
            if 20 < cv2.contourArea(cnt) < 6000:
                squares.append(cnt)
                
    return squares


def order_points(pts):
    # Reference: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    # the order of the list: top-left -> top->right -> bottom->right -> bottom->left
    ls = np.zeros((4,2),dtype="float32")
    s = pts.sum(axis=1)
    ls[0] = pts[np.argmin(s)]
    ls[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    ls[1] = pts[np.argmin(diff)]
    ls[3] = pts[np.argmax(diff)]
    return ls 

def homography(src, dst):
    A = []
    src = order_points(src)
    
    for i in range(len(dst)):
        x_w, y_w = src[i][0], src[i][1]
        x_c, y_c = dst[i][0], dst[i][1]
        A.append([x_w, y_w, 1, 0, 0, 0, -x_c * x_w, -x_c * y_w, -x_c])
        A.append([0, 0 , 0, x_w, y_w, 1, -y_c * x_w, -y_c * y_w, -y_c])
    
    A = np.array(A)
    U, D, V_t = np.linalg.svd(A) 
    h = V_t[-1, :] / V_t[-1, -1]
    H_mat = np.reshape(h, (3, 3))
    
    return H_mat



def get_points(t):
    array = []
    for i in t:
        array.append(i[0])
    return array


def rotation_and_translation_mat(K, H): 
    K_inv = np.linalg.inv(K)
    l_d = 1/(np.linalg.norm(K_inv @ H[:,0]) + np.linalg.norm(K_inv @ H[:,1]))/2

    B_hat = K_inv @ H
    if np.linalg.det(B_hat) > 0:
        # B = l_d * B_hat
        B = B_hat
    else:
        # B = -l_d * B_hat
        B = -B_hat
    
    b1=B[:,0]
    b2=B[:,1]
    b3=B[:,2]
    r1 = l_d * b1
    r2 = l_d * b2
    t = l_d * b3 
    r3 = np.cross(r1,r2)
    T = np.array([t]).T
    R = np.array([r1, r2, r3]).T
    # P = K @ np.stack((r1,r2,r3,t), axis=1)
    return R, T

def draw_cube(img, corner_pts):
    corner_pts = np.int32(corner_pts).reshape(-1,2)
    img = cv2.drawContours(img, [corner_pts[:4]], -1,(0,0,255), 3)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(corner_pts[i]), tuple(corner_pts[j]),(0,0,255), 3)
    img = cv2.drawContours(img, [corner_pts[4:]],-1,(0,0,255), 3)
    return img

def project_cube(H_mat, img):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html
    axis = np.float32([[0,0,0],
                              [0,cube_dim,0],
                              [cube_dim,cube_dim,0],
                              [cube_dim,0,0],
                              [0,0,-cube_dim],
                              [0,cube_dim,-cube_dim],
                              [cube_dim,cube_dim,-cube_dim],
                              [cube_dim,0,-cube_dim]])
    K = np.array([[1406.08415449821,0,0],
           [2.20679787308599, 1417.99930662800,0],
           [1014.13643417416, 566.347754321696,1]])
    # Remember to TRANSPOSE K
    K = K.T
    # Calculate the Rotation Matrix and Translation Vector instead of Projection Matrix
    R_mat, t_vector = rotation_and_translation_mat(K, H_mat)
    # Project 3D points to image plane
    proj_corner_pts, jacobian = cv2.projectPoints(axis, R_mat, t_vector, K, np.zeros((1, 4)))
    
    img = draw_cube(img, proj_corner_pts)

    
    return img

def main(video_dir):
    cap = cv2.VideoCapture(video_dir)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Impose_Cube.avi',fourcc, 20.0, (1152, 648))
    
    if (cap.isOpened()==False):
        print("Error")
    while cap.isOpened(): 
        ret, frame = cap.read()
        
        if ret == True:
            scale_percent = 60 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
            img_copy_for_cube = copy.deepcopy(frame)
            
            tag_contour_ls = find_tag(frame)
            for i in range(len(tag_contour_ls)):
                corner = get_points(list(tag_contour_ls[i]))
                H_mat = homography(cube_base, corner)
                img_copy_for_cube = project_cube(H_mat, img_copy_for_cube)
                
            frame = copy.deepcopy(img_copy_for_cube)
            out.write(frame)
            cv2.imshow('f', frame)  
            if cv2.waitKey(10) & 0xFF == ord("q"): 
                break 
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows() 



if __name__ == "__main__":
    print('Choose a video to Run: ')
    while True:
        print('Press 0 for Tag0.mp4, 1 for Tag1.mp4, 2 for Tag2.mp4, 3 for multipleTags.mp4 ')
        video_option = str(input())
        if video_option == '0' or video_option == '1' or video_option == '2' or video_option == '3':
            break
        else:    
            print("Sorry, Please Enter 0, 1, 2, or 3 to choose a video")

        
    video_option_dict = {'0': './Tag0.mp4', '1': './Tag1.mp4', '2': './Tag2.mp4', '3': './multipleTags.mp4'}
    main(video_option_dict[video_option])