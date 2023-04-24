Flappy GPU
===

This is just a clone of flappy bird but with mesh shaders for generating the geometry.

It's basically a mesh shader and texture array example in vulkan

There's no scoreboard/menu because I still need to look up how to do font rendering.

In the future if I come back to this the plan is to move the update logic completely to a compute shader
and just send a push constant of time difference to the gpu as a clock
and then any restart/etc. is just a shared memory boolean the compute shader
can write to on gameover
