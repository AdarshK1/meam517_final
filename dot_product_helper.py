import numpy as np

link_origin = np.array([0.0, 0.0, 0.0])
link_end = np.array([1.0, 0.0, -1.0])

obstacle_in_link = np.array([-0.5, 0, 0.5])

link_vect = link_end - link_origin
# norm of link vector
link_vect_norm = np.linalg.norm(link_vect)
# link unit vector
link_unit_vect = link_vect / link_vect_norm

# vector going from link to obstacle
link_to_obst_vect = obstacle_in_link - link_origin

projection = (link_unit_vect * link_to_obst_vect.dot(link_unit_vect))

obst_dist = link_to_obst_vect - projection

# this means that the obstacle is farther from the link than the length of the link
link_end_dist = np.linalg.norm(obstacle_in_link - link_end)
link_origin_dist = np.linalg.norm(obstacle_in_link - link_origin)

distance = np.linalg.norm(link_end_dist)

if link_origin_dist < distance:
    distance = link_origin_dist
if link_end_dist < distance:
    distance = link_end_dist


# distance = np.linalg.norm(obst_dist)

print("link_vect\t\t", link_vect, "\tmag:", np.linalg.norm(link_vect))
print("link_to_obst_vect\t", link_to_obst_vect, "\tmag:", np.linalg.norm(link_to_obst_vect))
print("projection", projection)
print("obst_dist", obst_dist)
print("\n")
print("Distance:", distance)
