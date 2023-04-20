package main

import "render"

import "core:fmt"

UniformData :: struct {
    translation: matrix[3, 3]f32, // 2d affine transformation
}

main :: proc() {
    render.global_render_init()
    defer render.global_render_destroy()

    fmt.println("hello flappy gpu!")
    render.hello()

    x := [2]f32{0.3, -0.2}
    x_expanded := [3]f32{x.x, x.y, 1}
    ud := UniformData{
        translation = matrix[3, 3]f32{
            1, 0, 0.3,
            0, 1, -0.2,
            0, 0, 1,
        },
    }

    result := ud.translation * x_expanded
    new_x := result.xy
    fmt.println(x, result, new_x)
}

GOAL := `

The purpose of this is to make a flappy bird clone. So the window should be phone vertical size / aspect ratio
Flappy bird only has 2 sets of columns, the ones you're going through and the other ones.
Once the closest columns go off screen to the left the new ones appear on the right (when right side of column crosses edge of screen reset to edge of other screen)

Maybe we can learn mmesh/task shaders? I'd like to put almost everything on compute shaders each frame

Couple plans could be:
1. Mesh shader that converts a single point to the 2 rectangles of the column?
2. 

1. Make a box
2. Make 4 boxes for the columns
3. Columns randomize center height and then they just go until offscreen
4. It'll essentially be a UBO for each point/column, yeah that makes sense



background -> fixed
2 points for the columns
1 point for the bird
foreground rectangles (ground) -> fixed at first

Counter (stb_truetype? look this up later)
`