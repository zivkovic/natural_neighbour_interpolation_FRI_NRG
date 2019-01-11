import matplotlib.pyplot as pl
import numpy as np
import scipy.spatial
import math
import pickle

from PIL import Image

input_folder = './input/'
output_folder = './output/'


def memoize_edges_for_region(f):
    memo = {}

    def helper(self, point):
        key = point['idx']
        if key not in memo:
            memo[key] = f(self, point)
        return memo[key]
    return helper


class NaturalNeighbourInterpolation:

    def __init__(self, file_name=None, number_of_sites=None):
        if file_name is not None and number_of_sites is None:
            self.read_from_pickle(file_name)
            self.add_clipping()
            self.create_voronoi()
        else:
            self.number_of_sites = number_of_sites
            self.sample_from_image(file_name)

    def add_clipping(self):
        bounding_box = np.array([0., self.width, 0., self.height])
        points_center = self.sites
        points_left = np.copy(points_center)
        points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
        points = np.append(points_center,
                           np.append(np.append(points_left,
                                               points_right,
                                               axis=0),
                                     np.append(points_down,
                                               points_up,
                                               axis=0),
                                     axis=0),
                           axis=0)

        point_colors = dict()
        n = len(self.sites)
        for idx in range(n):
            color = self.site_colors[idx]
            left_idx = n + idx
            right_idx = n * 2 + idx
            down_idx = n * 3 + idx
            up_idx = n * 4 + idx
            point_colors[idx] = color
            point_colors[left_idx] = color
            point_colors[right_idx] = color
            point_colors[down_idx] = color
            point_colors[up_idx] = color

        self.sites = points
        self.site_colors = point_colors

    def create_voronoi(self):
        self.voronoi = scipy.spatial.Voronoi(self.sites)

        # swap key/value in dict
        old_ridge_dict = dict(self.voronoi.ridge_dict)
        self.voronoi.ridge_dict.clear()
        for key, value in old_ridge_dict.items():
            self.voronoi.ridge_dict[tuple(value)] = key

        # create kdtree
        self.kdtree = scipy.spatial.cKDTree(self.sites)

        # create site dict
        self.sites_dict = {}
        #for idx, site in enumerate(self.sites):
        for idx in range(len(self.site_colors)):
            site = self.sites[idx]
            self.sites_dict[(int(site[0]), int(site[1]))] = self.site_colors[idx]

        #self.draw()

    def add_point(self, point):
        #print('Started processing point: ', point)
        int_point = (int(point[0]), int(point[1]))
        if int_point in self.sites_dict:
            color = self.sites_dict[int_point]
            return color
        #new_edges = []
        list_of_area_and_color = []
        closest_point = self.get_closest_point(point)
        perpendicular_bisector = self.get_perpendicular_bisector(closest_point['point'], point)
        e1, e2 = self.get_edges_crossing_bisector(closest_point, perpendicular_bisector, point)
        #new_edges.extend([e1['intersection'], e2['intersection']])
        list_of_area_and_color.append(self.calculate_area_taken((e1, e2), closest_point))

        final_edge = e1
        prev_point = closest_point
        loop_counter = 0
        while not self.edge_equals(e2, final_edge):
            if loop_counter > 20: return 0
            other_point = self.get_other_point(e2, prev_point)
            perpendicular_bisector = self.get_perpendicular_bisector(other_point['point'], point)
            e1, e2 = self.get_edges_crossing_bisector(other_point, perpendicular_bisector, point)
            #new_edges.extend([e1['intersection'], e2['intersection']])
            list_of_area_and_color.append(self.calculate_area_taken((e1, e2), other_point))
            prev_point = other_point
            loop_counter += 1

        return self.calculate_weighed_color(list_of_area_and_color)

    def get_closest_point(self, point):
        query_res = self.kdtree.query(point)
        return {
            'idx': query_res[1],
            'point': self.kdtree.data[query_res[1]]
        }

    # Cython code
    def get_perpendicular_bisector(self, a, b):
        return optimized.get_perpendicular_bisector(a[0], a[1], b[0], b[1])

    def get_edges_crossing_bisector(self, point, bisector, sort_point):
        edges = self.get_edges_for_region_with_bisector(point, bisector)

        if len(edges) != 2:
            raise Exception('Edges must be 2, were ' + str(len(edges)) + '!')

        ccw_value = self.ccw(edges[0]['intersection'][0], edges[0]['intersection'][1], edges[1]['intersection'][0], edges[1]['intersection'][1], sort_point[0], sort_point[1])
        if ccw_value >= 1:# is_ccw
            return edges[0], edges[1]
        return edges[1], edges[0]

    def calculate_area_taken(self, new_edge, point):
        stolen_area = self.calculate_area_for_stolen_polygon(new_edge, point)
        return [stolen_area, self.site_colors[point['idx']]]

    def get_vertices_for_region(self, point):
        region_idx = self.voronoi.point_region[point['idx']]
        region = self.voronoi.regions[region_idx]
        if -1 in region:
            raise Exception('-1 as vertexIdx!')
        vertices = [
            {
                'idx': vertex_idx,
                'vertex': self.voronoi.vertices[vertex_idx]
            } for vertex_idx in region
        ]

        return self.sort_vertices(vertices)

    def sort_vertices(self, vertices):
        # calculate centroid of the polygon
        n = len(vertices)  # of corners
        cx = float(sum(vertex['vertex'][0] for vertex in vertices)) / n
        cy = float(sum(vertex['vertex'][1] for vertex in vertices)) / n
        # create a new list of corners which includes angles
        corners_with_angles = []
        for vertex in vertices:
            x, y = vertex['vertex']
            an = (math.atan2(y - cy, x - cx) + 2.0 * math.pi) % (2.0 * math.pi)
            corners_with_angles.append((vertex, an))
        # sort it using the angles
        corners_with_angles.sort(key=lambda tup: tup[1])
        # return the sorted corners w/ angles removed
        return list(map(lambda x: x[0], corners_with_angles))

    # Cython code
    def calculate_area_for_stolen_polygon(self, new_edge, stolen_from_point):
        return optimized.calculate_area_for_stolen_polygon(self, new_edge, stolen_from_point)

    def get_other_point(self, edge, point):
        try:
            ridge = self.voronoi.ridge_dict[(edge['start']['idx'], edge['end']['idx'])]
        except KeyError:
            ridge = self.voronoi.ridge_dict[(edge['end']['idx'], edge['start']['idx'])]

        ridge_point_idx = ridge[1] if ridge[0] == point['idx'] else ridge[0]
        return {
            'idx': ridge_point_idx,
            'point': self.voronoi.points[ridge_point_idx]
        }

    def sample_random_sites(self):
        return np.c_[
            np.random.uniform(0, self.width, self.number_of_sites),
            np.random.uniform(0, self.height, self.number_of_sites)
        ], np.random.randint(0xFFFFFFFF, size=self.number_of_sites, dtype=np.uint32)

    def sample_from_image(self, image_name):
        image = Image.open(input_folder + image_name)
        pixels = np.array(image)
        self.width, self.height = image.size
        self.sites, _ = self.sample_random_sites()
        self.site_colors = [pixels[int(point[1])][int(point[0])] for point in self.sites]
        self.add_clipping()
        self.create_voronoi()

    def generate_image(self):
        pixels = []
        for y in np.linspace(0, self.height, self.height):
            print('Processing line: ', int(y), ' of ', self.height)
            x_row = []
            for x in np.linspace(0, self.width, self.width):
                try:
                    color = self.add_point((x, y))
                except Exception as e:
                    #print(e)
                    color = (0, 0, 0)
                x_row.append(color)
            pixels.append(x_row)

        print('Saving image.')
        # Convert the pixels into an array using numpy
        pixels = np.array(pixels, dtype=np.uint8)

        # Use PIL to create an image from the new array of pixels
        new_image = Image.fromarray(pixels)
        new_image.save(output_folder + 'output.png')

        print('Done!')

    def get_edges_for_region(self, point):
        vertices = self.get_vertices_for_region(point)
        n = len(vertices)
        edges = []
        for i in range(n):
            first = vertices[i]
            second = vertices[(i + 1) % n]

            edges.append({
                'start': first,
                'end': second
            })
        return edges

    def get_edges_for_region_with_bisector(self, point, bisector):
        edges = self.get_edges_for_region(point)
        new_edges = []
        for edge in edges:
            intersection = None
            if bisector is not None:
                first = edge['start']
                second = edge['end']
                intersection = self.get_line_intersection(
                    first['vertex'][0], first['vertex'][1], second['vertex'][0], second['vertex'][1],
                    bisector[0][0], bisector[0][1], bisector[1][0], bisector[1][1]
                )
            edge['intersection'] = intersection

            if edge['intersection'] is not None:
                new_edges.append(edge)
        return new_edges

    def draw(self, **kw):
        #sp.spatial.voronoi_plot_2d(self.voronoi, show_vertices=False, point_size=2)
        #
        vor = self.voronoi
        fig = pl.figure(figsize=(16, 16))
        ax = fig.gca()
        from matplotlib.collections import LineCollection

        if vor.points.shape[1] != 2:
            raise ValueError("Voronoi diagram is not 2-D")

        if kw.get('show_points', True):
            point_size = kw.get('point_size', None)
            ax.plot(vor.points[:, 0], vor.points[:, 1], '.', markersize=point_size)
        if kw.get('show_vertices', True):
            ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')

        line_colors = kw.get('line_colors', 'k')
        line_width = kw.get('line_width', 1.0)
        line_alpha = kw.get('line_alpha', 1.0)

        center = vor.points.mean(axis=0)
        ptp_bound = vor.points.ptp(axis=0)

        finite_segments = []
        infinite_segments = []
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[i] + direction * ptp_bound.max()

                infinite_segments.append([vor.vertices[i], far_point])

        ax.add_collection(LineCollection(finite_segments,
                                         colors=line_colors,
                                         lw=line_width,
                                         alpha=line_alpha,
                                         linestyle='solid'))
        ax.add_collection(LineCollection(infinite_segments,
                                         colors=line_colors,
                                         lw=line_width,
                                         alpha=line_alpha,
                                         linestyle='dashed'))

        margin = 0.1 * vor.points.ptp(axis=0)
        xy_min = vor.points.min(axis=0) - margin
        xy_max = vor.points.max(axis=0) + margin
        #ax.set_xlim(xy_min[0], xy_max[0])
        #ax.set_ylim(xy_min[1], xy_max[1])

        pl.show()
        #pl.savefig(output_folder + 'voronoi.png')

    def debug_draw(self, point, edges):
        fig = pl.figure(figsize=(16, 16))
        ax = fig.gca()
        """
        # Plot initial points
        ax.plot(self.voronoi.points[:, 0], self.voronoi.points[:, 1], 'b.')
        # Plot ridges points
        for region in self.voronoi.regions:
            vertices = self.voronoi.vertices[region, :]
            ax.plot(vertices[:, 0], vertices[:, 1], 'go')
        # Plot ridges
        for region in self.voronoi.regions:
            if len(region) == 0: continue;
            vertices = self.voronoi.vertices[region + [region[0]], :]
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
        """
        from matplotlib.collections import LineCollection

        line_colors = 'k'
        line_width = 1.0
        line_alpha = 1.0

        center = self.voronoi.points.mean(axis=0)
        ptp_bound = self.voronoi.points.ptp(axis=0)

        ax.plot([point[0]], [point[1]], 'yo')

        ax.plot(self.voronoi.points[:, 0], self.voronoi.points[:, 1], '.', markersize=None)# points
        ax.plot(self.voronoi.vertices[:, 0], self.voronoi.vertices[:, 1], '.')# vertices

        finite_segments = []
        infinite_segments = []
        for pointidx, simplex in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(self.voronoi.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = self.voronoi.points[pointidx[1]] - self.voronoi.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = self.voronoi.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.voronoi.vertices[i] + direction * ptp_bound.max()

                infinite_segments.append([self.voronoi.vertices[i], far_point])

        ax.add_collection(LineCollection(finite_segments,
                                         colors=line_colors,
                                         lw=line_width,
                                         alpha=line_alpha,
                                         linestyle='solid'))
        ax.add_collection(LineCollection(infinite_segments,
                                         colors=line_colors,
                                         lw=line_width,
                                         alpha=line_alpha,
                                         linestyle='dashed'))

        x_coords = []
        y_coords = []
        for idx, edge in enumerate(edges):
            if idx == 0:
                x_coords.append(edge[0][0])
                y_coords.append(edge[0][1])
            x_coords.append(edge[1][0])
            y_coords.append(edge[1][1])
            """
            if idx == 0:
                x_coords.append(edge['start']['vertex'][0])
                y_coords.append(edge['start']['vertex'][1])
            x_coords.append(edge['end']['vertex'][0])
            y_coords.append(edge['end']['vertex'][1])
            """
            ax.arrow(edge[0][0], edge[0][1], edge[1][0] - edge[0][0], edge[1][1] - edge[0][1], color='red', head_width=0.5)#2

        #ax.plot(x_coords, y_coords, 'r-')

        margin = 0.1 * self.voronoi.points.ptp(axis=0)
        xy_min = self.voronoi.points.min(axis=0) - margin
        xy_max = self.voronoi.points.max(axis=0) + margin
        ax.set_xlim(xy_min[0], xy_max[0])
        ax.set_ylim(xy_min[1], xy_max[1])

        pl.show()
        pl.savefig(output_folder + 'debug.png')

    # Cython code
    @staticmethod
    def calculate_weighed_color(list_area_and_color):
        return optimized.calculate_weighed_color(list_area_and_color, len(list_area_and_color))

    @staticmethod
    def edge_equals(a, b):
        dx = a['intersection'][0] - b['intersection'][0]
        dy = a['intersection'][1] - b['intersection'][1]
        return - 0.000001 < dx < 0.000001 and - 0.000001 < dy < 0.000001

    # Cython code
    def ccw(self, Ax, Ay, Bx, By, Cx, Cy):
        return optimized.ccw(Ax, Ay, Bx, By, Cx, Cy)

    # Cython code
    def get_line_intersection(self, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy):
        return optimized.get_line_intersection(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy)

    def write_to_pickle(self):
        print('Started saving to pickle!')
        obj = {
            'size': {
                'width': self.width,
                'height': self.height
            },
            'sites': self.sites,
            'site_colors': self.site_colors

        }
        pickle.dump(obj, open(output_folder + 'dump.pickle', 'wb'))
        print('Finished saving to pickle!')

    def read_from_pickle(self, file_name):
        print('Started loading from pickle!')
        obj = pickle.load(open(input_folder + file_name, 'rb'))
        self.width = obj['size']['width']
        self.height = obj['size']['height']
        self.sites = obj['sites']
        self.site_colors = obj['site_colors']
        print('Finished loading from pickle!')


if __name__ == '__main__':
    NaturalNeighbourInterpolation.get_edges_for_region = memoize_edges_for_region(NaturalNeighbourInterpolation.get_edges_for_region)

    import pyximport

    pyximport.install(setup_args={
        'include_dirs': np.get_include()
    })

    import optimized

    # Run random points on 300x300 image with 1000 samples
    #nni = NaturalNeighbourInterpolation('300x300.jpg', 1000)
    #nni.generate_image()

    # Run random points on 300x300 image with 10000 samples
    #nni = NaturalNeighbourInterpolation('300x300.jpg', 10000)
    #nni.generate_image()

    # Run random points on 1024x1024 image with 100000 samples
    nni = NaturalNeighbourInterpolation('1024x1024.jpg', 100000)
    nni.generate_image()

    # Save dump of sampled data
    #nni.write_to_pickle()

    # Read from dump of sampled data
    #nni = NaturalNeighbourInterpolation('dump.pickle')

