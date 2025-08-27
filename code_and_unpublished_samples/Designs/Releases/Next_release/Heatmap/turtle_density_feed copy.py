
from turtle import *
import turtle

# pensize(3)
# speed(0)
# bgcolor("white")
# color("black")

# for i in range(255):
#     for j in range(4):
#         circle(200-i, steps = 200-i)


import turtle as t
from turtle import Screen
import random as rn



#torus ish
# t.colormode(255)

# def rand_color():
#     r = rn.randint(0,255)
#     g = rn.randint(0,255)
#     b = rn.randint(0,255)
#     return (r, g, b)

# def til_circle(angle):
#     for i in range(int(360/angle)):
#         t.pensize(2)
#         t.speed('fastest')
#         t.color('black')
#         t.circle(200)
#         t.setheading(t.heading() + angle)

# til_circle(1)

# screen = Screen()
# screen.exitonclick()
 
# t.speed('fastest')

# for i in range(10):
#     for j in range(4):
#         t.right(90)
#         t.circle(200-15*i, steps = 90)
#         t.left(90)
#         t.circle(200-15*i, steps = 90)
#         t.right(180)
#         t.circle(50, 24)


# t.speed('fastest')


# # for steps in range(200):
# #     forward(steps)
# #     right(45)


# #max size for getscreen is like 200

# for I in range(100):
#     forward(200)
#     right(150)
#     forward(200)
#     left(20)


# # forward(100)
# # left(150)
# # forward(100)


# # Grab the canvas from the screen
# t.hideturtle()


# screen = t.Screen()

# screen.update()




# # Grab the canvas

# canvas = screen.getcanvas()

# # Larger or custom dimensions (in postscript points, 1 point ~ 1/72 inch)
# canvas.postscript(
#     file="/Users/henryschnieders/Documents/ScriptedStyles/Designs/Releases/Next_release/Heatmap/Data/turtle_density_feed.eps",
#     colormode="color",
#     x=0,
#     y=0,
#     width=800,      # the drawing area width in points
#     height=800,     # the drawing area height in points
#     pagewidth=800,  # the page width in points
#     pageheight=800  # the page height in points
# )

import turtle as t
from turtle import Screen

# Create the screen and turtle
screen = Screen()
screen.setup(width=800, height=800)  # sets the window size
t.speed('fastest')

# Example drawing

# for steps in range(200):
#     forward(steps)
#     right(45)



# import turtle

import math



def koch_curve(length, depth):
    """Recursively draws one segment of a Koch curve."""
    if depth == 0:
        t.forward(length)
    else:
        length /= 3.0
        koch_curve(length, depth - 1)
        t.left(60)
        koch_curve(length, depth - 1)
        t.right(120)
        koch_curve(length, depth - 1)
        t.left(60)
        koch_curve(length, depth - 1)

def koch_snowflake(length, depth):
    """Draws a full Koch snowflake (3 sides) using the Koch curve."""
    for _ in range(3):
        koch_curve(length, depth)
        t.right(120)

def draw_centered_snowflake(length, depth):
    """
    Positions the turtle so that an equilateral triangle of side `length` 
    (the base of our snowflake) is centered at (0, 0), then draws the snowflake.
    """
    # Compute starting position:
    # We want the centroid (x + L/2, y - L*sqrt(3)/6) to be (0, 0)
    start_x = -length / 2
    start_y = length * math.sqrt(3) / 6
    t.penup()
    t.goto(start_x, start_y)
    t.setheading(0)  # Ensure we're facing right
    t.pendown()
    koch_snowflake(length, depth)

# Set up the turtle
t.speed('fastest')
t.hideturtle()

# Draw three snowflakes of different sizes, all centered at (0, 0)
draw_centered_snowflake(400, 3)      # Largest snowflake
draw_centered_snowflake(400/2, 3)      # Smaller one
draw_centered_snowflake(400*(1/4), 3)  # Medium-sized one





t.hideturtle()


screen.update()

# Get the canvas and compute the bounding box of all items
canvas = screen.getcanvas()
bbox = canvas.bbox("all")  # returns (xmin, ymin, xmax, ymax)
print("Bounding Box:", bbox)

if bbox:
    x, y, x2, y2 = bbox
    width = x2 - x
    height = y2 - y

    canvas.postscript(
        file="/Users/henryschnieders/Documents/ScriptedStyles/Designs/Releases/Next_release/Heatmap/Data/turtle_density_feed.eps",
        colormode="color",
        x=x,
        y=y,
        width=width,
        height=height,
        pagewidth=width,
        pageheight=height
    )
