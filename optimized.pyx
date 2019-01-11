# cython: language_level=3
# cython: boundscheck=False

import cython
from libc.math cimport abs, round
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.cdivision(True)
def get_line_intersection(double Ax, double Ay, double Bx, double By, double Cx, double Cy, double Dx, double Dy):
    cdef double Bx_Ax = Bx - Ax
    cdef double By_Ay = By - Ay
    cdef double Dx_Cx = Dx - Cx
    cdef double Dy_Cy = Dy - Cy
    cdef double determinant = (-Dx_Cx * By_Ay + Bx_Ax * Dy_Cy)
    if abs(determinant) < 1e-20:
        return None
    cdef double s = (-By_Ay * (Ax - Cx) + Bx_Ax * (Ay - Cy)) / determinant
    cdef double t = (Dx_Cx * (Ay - Cy) - Dy_Cy * (Ax - Cx)) / determinant
    cdef double output[2]
    if s >= 0 and s <= 1 and t >= 0 and t <= 1:
        output[0] = Ax + (t * Bx_Ax)
        output[1] = Ay + (t * By_Ay)
        return output
    return None

def ccw(double Ax, double Ay, double Bx, double By, double Cx, double Cy):
    cdef double area2 = (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)
    if area2 < 0:
        return -1
    elif area2 > 0:
        return 1
    return 0

cdef get_center_on_line(double startX, double startY, double endX, double endY):
    cdef double x = (startX + endX) / 2.0
    cdef double y = (startY + endY) / 2.0
    return x, y

@cython.cdivision(True)
def get_perpendicular_bisector(double Ax, double Ay, double Bx, double By):
    cdef (double, double) center = get_center_on_line(Ax, Ay, Bx, By)
    cdef double slope = (By - Ay) / (Bx - Ax)
    cdef double perpendicular_line_slope = -1 / slope
    cdef double output[2][2]
    output[0][0] = 10000
    output[0][1] = (perpendicular_line_slope * (10000 - center[0])) + center[1]
    output[1][0] =- 10000
    output[1][1] = (perpendicular_line_slope * (-10000 - center[0])) + center[1]
    return output

#https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates/30408825#30408825
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef double calculate_area_from_vertices(list vertices, int N):
    cdef int i
    cdef int j
    cdef double area = 0.0
    for i in range(N):
        j = (i + 1) % N
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return 0.5 * abs(area)

def calculate_weighed_color(list list_area_and_color not None, int n):
    cdef double full_area = 0
    cdef int i
    for i in range(n):
        full_area += list_area_and_color[i][0]
    #full_area = sum([t[0] for t in list_area_and_color])
    cdef int r = 0
    cdef int g = 0
    cdef int b = 0

    cdef list t
    for i in range(n):
        t = list_area_and_color[i]
        percentage = t[0] / full_area
        r += t[1][0] * percentage
        g += t[1][1] * percentage
        b += t[1][2] * percentage
    #return int(color[0]), int(color[1]), int(color[2])
    cdef double output[3]
    output[0] = round(r)
    output[1] = round(g)
    output[2] = round(b)
    return output
    #return round(r), round(g), round(b)

def calculate_area_for_stolen_polygon(self, new_edge, stolen_from_point):
    stolen_from_edges = self.get_edges_for_region(stolen_from_point)
    vertices = [new_edge[0]['intersection']]
    cdef double[2] prev_vertex = new_edge[1]['intersection']
    cdef int i = 0
    prev_edge = new_edge[1]
    cdef int final_edge_idx = new_edge[0]['start']['idx']
    cdef bint first_done = False
    cdef int stolen_from_edges_len = len(stolen_from_edges)
    while True:
        edge = stolen_from_edges[i % stolen_from_edges_len]

        if first_done:
            if edge['start']['idx'] == final_edge_idx:
                vertices.append(prev_vertex)
                break
            vertices.append(prev_vertex)
            prev_vertex = edge['end']['vertex']
            prev_edge = edge
        else:
            if prev_edge['start']['idx'] == edge['start']['idx'] and prev_edge['end']['idx'] == edge['end']['idx']:
                vertices.append(prev_vertex)
                prev_vertex = edge['end']['vertex']
                prev_edge = edge
                first_done = True
            elif prev_edge['end']['idx'] == edge['start']['idx'] and prev_edge['start']['idx'] == edge['end']['idx']:
                vertices.append(prev_vertex)
                prev_vertex = edge['start']['vertex']
                prev_edge = edge
                first_done = True
        i += 1
    return calculate_area_from_vertices(vertices, len(vertices))