#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:50:59 2021

@author: bernard
"""
import numpy as np 
import cv2 
import copy

dim = 80
pic = np.array([
		[0, 0],
		[dim-1, 0],
		[dim-1, dim-1],
		[0, dim-1]], dtype = "float32")

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
        cnt = cv2.approxPolyDP(cnt, 0.015*cnt_len, True) # 0.02 better than 0.01
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


def warpPerspective(H_mat, src_img, dest_img, dest_pts):
    H_inv = np.linalg.pinv(H_mat)
    H_inv = H_inv / H_inv[2][2]
    
    dest_img_copy = copy.deepcopy(dest_img)
    src_img_dim = src_img.shape
    col_min, row_min = np.min(dest_pts, axis=0) 
    col_max, row_max = np.max(dest_pts, axis=0)
    
    for y_ind in range(int(row_min), int(row_max)):   
        for x_ind in range(int(col_min), int(col_max)):
            dest_pt = np.float32([x_ind, y_ind, 1]).T
            src_pt = H_inv @ dest_pt
            src_pt = (src_pt/src_pt[2]).astype(int)
            if -1 < src_pt[1] < src_img_dim[0] and -1 < src_pt[0] < src_img_dim[1]:
                    dest_img_copy[y_ind, x_ind] = src_img[src_pt[1], src_pt[0]]
    
    
    return dest_img_copy


def warped_img(src_img, dest_img, src_pts, dest_pts):
    # src_pts = order_points(src_pts[:,0])
    H_mat = homography(src_pts, dest_pts)
    warped = warpPerspective(H_mat, src_img, dest_img, dest_pts)
    return warped





def get_warped_tag(img, tag_ls):
    warped_tag = []
    dest_img = copy.deepcopy(img[:dim, :dim])
    dest_img[:, :] = 0
    if len(tag_ls) > 0:
        for key in tag_ls:
            warp = warped_img(img, dest_img, np.float32(order_points(key[:,0])), pic)
            warped_tag.append(warp)
    return warped_tag

def is_cell_white(cell):
    threshold = 200
    cell_to_gray = cv2.cvtColor((cell), cv2.COLOR_BGR2GRAY) if len(cell.shape) > 2 else cell
    return 1 if (np.mean(cell_to_gray) >= threshold) else 0

def get_ar_tag_id(tag_image):
    _, binary = cv2.threshold(tag_image, 200, 255, cv2.THRESH_BINARY_INV)
    tag_corners_map = {}
    tag_corners_map["TL"] = tag_image[20:30, 20:30] 
    tag_corners_map["TR"] = tag_image[20:30, 50:60] 
    tag_corners_map["BR"] = tag_image[50:60, 50:60] 
    tag_corners_map["BL"] = tag_image[50:60, 20:30] 
    
    inner_corners_map = {}
    inner_corners_map["TL"] = tag_image[30:40, 30:40]
    inner_corners_map["TR"] = tag_image[30:40, 40:50] 
    inner_corners_map["BR"] = tag_image[40:50, 40:50] 
    inner_corners_map["BL"] = tag_image[40:50, 30:40] 
    
    white_cell_corner = ''
    for cell_key in tag_corners_map:
        if is_cell_white(tag_corners_map[cell_key]):
            white_cell_corner = cell_key
            break
    if white_cell_corner == '':
        return None
    
    print('White Cell Corner: ', white_cell_corner)
    # id_number = [(is_cell_white(inner_corners_map[cell_key])) for cell_key in inner_corners_map]
    # print('id_number', id_number)
    # re_orient_action_map = {'BL': [3, 0, 1, 2], 'TL': [2, 3, 0, 1], 'TR': [1, 2, 3, 0], 'BR': [0, 1, 2, 3]}

    # tag_id = 0
    # for index, swap_ind in enumerate(re_orient_action_map[white_cell_corner]):
    #     tag_id = tag_id + id_number[swap_ind] * np.power(2, (index))
        
    return white_cell_corner

def rotate_img(img,orientation):
    if orientation == 'TR':
        rotated_img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 'TL':
        rotated_img = cv2.rotate(img,cv2.ROTATE_180)
    elif orientation == 'BL':
        rotated_img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated_img = img

    return rotated_img


def get_points(t):
    # array = []
    # for i in t:
    #     array.append(i[0])
    ls = list(i[0] for i in t)
    return ls



def main(video_dir):
    cap = cv2.VideoCapture(video_dir)
    testudo_img = cv2.imread('./testudo.png')
    testudo_img = cv2.resize(testudo_img, (dim, dim))
    testudo_width, testudo_height, channel = testudo_img.shape
    testudo_corners = np.float32([[0, 0], 
                                 [testudo_width-1, 0],
                                 [testudo_width-1, testudo_height-1],
                                 [0, testudo_height-1]])
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Impose_Testudo.avi',fourcc, 20.0, (1152, 648))
    
    if (cap.isOpened()==False):
        print("Error")
    while cap.isOpened(): 
        ret, frame = cap.read()
        
        if ret == True:
            scale_percent = 60 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
            
            # img_copy_for_id = copy.deepcopy(frame)
            img_copy_for_testudo = copy.deepcopy(frame)
            
            tag_contour_ls = find_tag(frame) 
            
            # cv2.drawContours(frame, tag_contour_ls, -1, (0,255,0), thickness = 2) 
            
            tag_img = get_warped_tag(frame, tag_contour_ls)
            
            # tag_num = 1
            if len(tag_contour_ls) > 0:
                for i in range(len(tag_img)):
                    corner = get_points(tag_contour_ls[i])
                    # tag_num += 1
                    white_corner = get_ar_tag_id(tag_img[i])
                    # print(tag_id)
                    
                    testudo_img = rotate_img(testudo_img, white_corner)
                    
                    H_testudo = homography(testudo_corners, corner)
                    
                    # cv2.fillConvexPoly(frame, np.int32(corner), 0, 16)
                    # img_copy_for_testudo = cv2.warpPerspective(testudo_img, H_testudo, (frame.shape[1], frame.shape[0]))
                    img_copy_for_testudo = warpPerspective(H_testudo, testudo_img, img_copy_for_testudo, corner) 

                
            frame = copy.deepcopy(img_copy_for_testudo)
            out.write(frame)
            # cv2.drawContours(frame, tag_contour_ls, -1, (0,255,0), thickness = 2) 
            cv2.imshow('f', frame)  
            if cv2.waitKey(30) & 0xFF == ord("q"): 
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