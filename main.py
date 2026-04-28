import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan)

width, height = 800, 600
pixels = ti.Vector.field(3, dtype=float, shape=(width, height))

# 光照参数
Ka = ti.field(float, shape=())
Kd = ti.field(float, shape=())
Ks = ti.field(float, shape=())
Shininess = ti.field(float, shape=())

Ka[None] = 0.2
Kd[None] = 0.7
Ks[None] = 0.5
Shininess[None] = 32.0

# 场景
cam_pos     = tm.vec3(0, 0, 5)
light_pos   = tm.vec3(2, 3, 4)
light_color = tm.vec3(1, 1, 1)
bg_color    = tm.vec3(0.0, 0.2, 0.3)

sphere_center = tm.vec3(-1.2, -0.2, 0)
sphere_radius = 1.2
sphere_color  = tm.vec3(0.8, 0.1, 0.1)

cone_tip    = tm.vec3(1.2, 1.2, 0)
cone_base_y = -1.4
cone_radius = 1.2
cone_color  = tm.vec3(0.6, 0.2, 0.8)

# ------------------------
# 球体求交
# ------------------------
@ti.func
def hit_sphere(ro, rd):
    oc = ro - sphere_center
    a = tm.dot(rd, rd)
    b = 2.0 * tm.dot(oc, rd)
    c = tm.dot(oc, oc) - sphere_radius * sphere_radius
    disc = b*b - 4*a*c
    t = -1.0
    if disc >= 0:
        t = (-b - tm.sqrt(disc)) / (2.0 * a)
    return t

# ------------------------
# 圆锥求交
# ------------------------
@ti.func
def hit_cone(ro, rd):
    t = -1.0
    H = cone_tip.y - cone_base_y
    tanR = cone_radius / H

    ox = ro.x - cone_tip.x
    oy = ro.y - cone_tip.y
    oz = ro.z - cone_tip.z

    a = rd.x**2 + rd.z**2 - (tanR**2)*rd.y**2
    b = 2 * (ox*rd.x + oz*rd.z - (tanR**2)*oy*rd.y)
    c = ox**2 + oz**2 - (tanR**2)*oy**2

    disc = b*b - 4*a*c
    if disc >= 0 and a != 0:
        t0 = (-b - tm.sqrt(disc)) / (2*a)
        hit_y = ro.y + rd.y * t0
        if t0 > 0 and cone_base_y <= hit_y <= cone_tip.y:
            t = t0
    return t

# ------------------------
# 阴影检测
# ------------------------
@ti.func
def in_shadow(pos):
    shadow_ray_o = pos + 1e-4 * tm.normalize(light_pos - pos)
    shadow_ray_d = tm.normalize(light_pos - pos)
    t1 = hit_sphere(shadow_ray_o, shadow_ray_d)
    t2 = hit_cone(shadow_ray_o, shadow_ray_d)
    min_t = 1e9
    if t1 > 1e-4:
        min_t = ti.min(min_t, t1)
    if t2 > 1e-4:
        min_t = ti.min(min_t, t2)
    dist = tm.distance(pos, light_pos)
    return min_t < dist

# ------------------------
# 法向量（已修复：无if内return）
# ------------------------
@ti.func
def get_normal(pos, is_sphere):
    n = tm.vec3(0.0)
    if is_sphere:
        n = pos - sphere_center
    else:
        dx = pos.x - cone_tip.x
        dz = pos.z - cone_tip.z
        n = tm.vec3(dx, 0.02, dz)
    return tm.normalize(n)

# ------------------------
# Blinn-Phong 光照（已修复）
# ------------------------
@ti.func
def blinn_phong_shade(pos, normal, obj_color, shadow):
    N = normal
    L = tm.normalize(light_pos - pos)
    V = tm.normalize(cam_pos - pos)
    H = tm.normalize(L + V)

    ambient = Ka[None] * light_color * obj_color
    diff = ti.max(0.0, tm.dot(N, L))
    diffuse = Kd[None] * diff * light_color * obj_color
    spec = ti.max(0.0, tm.dot(N, H)) ** Shininess[None]
    specular = Ks[None] * spec * light_color

    final = ambient + diffuse + specular
    if shadow:
        final = ambient
    return tm.clamp(final, 0.0, 1.0)

# ------------------------
# 渲染主函数
# ------------------------
@ti.kernel
def render():
    for i, j in pixels:
        u = (i / width) * 2.0 - 1.0
        v = (j / height) * 2.0 - 1.0
        u *= width / height
        rd = tm.normalize(tm.vec3(u, v, -1))

        ts = hit_sphere(cam_pos, rd)
        tc = hit_cone(cam_pos, rd)

        t_min = -1.0
        is_sphere = False
        if ts > 1e-4 and (tc < 1e-4 or ts < tc):
            t_min = ts
            is_sphere = True
        elif tc > 1e-4:
            t_min = tc
            is_sphere = False

        if t_min > 1e-4:
            pos = cam_pos + rd * t_min
            n = get_normal(pos, is_sphere)
            clr = sphere_color if is_sphere else cone_color
            shadow = in_shadow(pos)
            pixels[i, j] = blinn_phong_shade(pos, n, clr, shadow)
        else:
            pixels[i, j] = bg_color

# ------------------------
# UI 窗口
# ------------------------
window = ti.ui.Window("Phong + Blinn-Phong + Shadow", (width, height))
canvas = window.get_canvas()
gui = window.get_gui()

while window.running:
    with gui.sub_window("参数面板", 0.03, 0.03, 0.28, 0.4):
        Ka[None] = gui.slider_float("Ka", Ka[None], 0.0, 1.0)
        Kd[None] = gui.slider_float("Kd", Kd[None], 0.0, 1.0)
        Ks[None] = gui.slider_float("Ks", Ks[None], 0.0, 1.0)
        Shininess[None] = gui.slider_float("Shininess", Shininess[None], 1.0, 128.0)
    render()
    canvas.set_image(pixels)
    window.show()