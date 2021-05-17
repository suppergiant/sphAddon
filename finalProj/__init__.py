#====================== BEGIN GPL LICENSE BLOCK ======================
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
#======================= END GPL LICENSE BLOCK ========================

bl_info = {
    "name": "SPH Particles",
    "author": "Dalton Durant(suppergiant)",
    "version": (1, 0, 0),
    "blender": (2, 83, 0),
    "location": "View3D",
    "description": "A simple particle handler using SPH physics",
    "warning": "",  # used for warning icon and text in addons panel
    "category": "Generic"
}

import bpy
import numpy as np

class Properties(bpy.types.PropertyGroup):
    "Properties"
    num_particles : bpy.props.IntProperty(
        name = 'Number of Particles',
        description = "How many particles you want to simulate.",
        default = 100,
        min = 10
    )
    size : bpy.props.FloatProperty(
        name = 'Particle Size',
        description = "The size of the particle",
        default = 0.1,
        min = 0.001
    )
    fluid_mass : bpy.props.FloatProperty(
        name = 'Total Fluid Mass',
        description = "The total mass of all particles combined",
        default = 1.0,
        min = 0.01
    )
    gravity : bpy.props.FloatProperty(
        name = 'Gravity',
        description = "The gravity acting in the world z-axis",
        default = -9.81
    )
    collisions : bpy.props.BoolProperty(
        name = 'Add Fluid Collsions',
        description = "Can add fluid collisons from Blender's builtin collison handler",
        default = False,
    )
    fluid_radius : bpy.props.FloatProperty(
        name = 'Fluid Radius',
        description = "Distance the fluid particles collide with each other",
        default = 1.0,
        min = 0.0
    )
    fluid_viscocity : bpy.props.FloatProperty(
        name = 'Viscocity Factor',
        description = "The factor of viscocity controling the fluid",
        default = 0.5,
        min = 0.0,
        max = 2.0
    )
    display : bpy.props.BoolProperty(
        name = 'Display Velocity',
        description = "Draw velocity data as color",
        default = False,
    )
    push_pull : bpy.props.FloatProperty(
        name = 'Attraction Force',
        description = "How much the particles want to move away or move close to each other",
        default = 0.005,
        min = -0.10,
        max = 1.0
    )
    push_pull_scale : bpy.props.FloatProperty(
        name = 'Attract Force Factor',
        description = "If pull, then this is intesnity of push. If push, then this is intensity of pull.",
        default = 0.01,
        min = 0.0,
        max = 1.0
    )
    source : bpy.props.BoolProperty(
        name = 'Hide Source',
        description = "Hides the source object of the particles but not the particles themselves",
        default = False,
    )
    lifetime : bpy.props.IntProperty(
        name = 'Particle Lifetime',
        description = "How many frames until each spawned particle dies",
        default = 500,
        min = 10
    )
    
    
class SPH_PT_Panel(bpy.types.Panel):
    bl_idname = "SPH_PT_Panel"
    bl_label = "SPH Addon"
    bl_category = "SPH Addon"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        sphtool = scene.sph_tool
        
        layout.prop(sphtool, "num_particles")
        layout.prop(sphtool, "size")
        layout.prop(sphtool, "fluid_mass")
        layout.prop(sphtool, "gravity")
        layout.prop(sphtool, "collisions")
        if (sphtool.collisions == True):
            layout.prop(sphtool, "fluid_radius")
            layout.prop(sphtool, "fluid_viscocity")
        layout.prop(sphtool, "display")
        layout.prop(sphtool, "push_pull")
        layout.prop(sphtool, "push_pull_scale")
        layout.prop(sphtool, "source")
        layout.prop(sphtool, "lifetime")
        
        row = layout.row()
        
        row.operator("view3d.sph_control", text = "Update")
        
        
class SPH_OT_Operator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "view3d.sph_control"
    bl_label = "SPH Controller"
    bl_description = "An SPH method for controlling particles"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        "Get Properties"
        scene = context.scene
        sphtool = scene.sph_tool
        N = sphtool.num_particles
        size = sphtool.size
        bool_collisions = sphtool.collisions
        R = sphtool.fluid_radius
        viscocity = sphtool.fluid_viscocity
        life = sphtool.lifetime
    
        # Prepare particle system
        #set domain as an object named "Cube" 
        object = bpy.data.objects["Source"]
        #remove all particle systems from the current domain if they exist
        bpy.ops.object.particle_system_remove()
        #delete all the previous bakes if there were any
        bpy.ops.ptcache.free_bake_all()
        #create a new particle system for this domain
        object.modifiers.new("ParticleSystem", 'PARTICLE_SYSTEM')
        #number of particles
        object.particle_systems[0].settings.count = N
        #make all particles appear and disappear at the same time
        object.particle_systems[0].settings.frame_start = 1
        object.particle_systems[0].settings.frame_end = 10
        #set how many frames the particles live for
        object.particle_systems[0].settings.lifetime = life
        object.particle_systems[0].settings.emit_from = 'VOLUME'
        object.particle_systems[0].settings.distribution = 'RAND'
        
        if (sphtool.collisions == True):
            object.particle_systems[0].settings.physics_type = 'FLUID'
            object.particle_systems[0].settings.fluid.repulsion = 1.0
            object.particle_systems[0].settings.fluid.stiff_viscosity = viscocity
            object.particle_systems[0].settings.fluid.fluid_radius = R
            
        if (sphtool.display == True):
            object.particle_systems[0].settings.display_color = 'VELOCITY'
            
        elif (sphtool.display == False):
            object.particle_systems[0].settings.display_method = 'RENDER'
            
        if (sphtool.source == True):
            #make source invisible
            object.show_instancer_for_viewport = False
        elif (sphtool.source == False):
            #make source visible
            object.show_instancer_for_viewport = True
            
        #object.particle_systems[0].settings.render_type = 'OBJECT'
        #object.particle_systems[0].settings.instance_object = bpy.data.objects["Icosphere"]
        object.particle_systems[0].settings.particle_size = size
        object.particle_systems[0].settings.display_size = size
        object.particle_systems[0].settings.normal_factor = 0.0
        object.particle_systems[0].settings.effector_weights.all = 0.0
        object.particle_systems[0].settings.effector_weights.gravity = 0

        object.particle_systems[0].settings.use_size_deflect = True

        #clear the post frame handler
        bpy.app.handlers.frame_change_post.clear()
        
        #run the function on each frame
        bpy.app.handlers.frame_change_post.append(main)

        # Update to a frame where particles are updated
        bpy.context.scene.frame_current = bpy.context.scene.frame_start + 1
        return {'FINISHED'}


def main(self, context):
    "Get Properties"
    scene = context.scene
    sphtool = scene.sph_tool
    IN = sphtool.push_pull
    OUT = -1 * sphtool.push_pull_scale
    N = sphtool.num_particles # number of particles
    M         = sphtool.fluid_mass     # total fluid mass
    G = np.array([0.0, 0.0, sphtool.gravity]) # external (gravitational) forces
    
    # Simulation parameters
    h = 0.004 / np.sqrt(N / 1000)                   # kernel radius
    WPOLY_CONST = 315.0 / (64.0 * np.pi * h**9.0)   # constant component of poly kernel 
    gradWPOLY_CONST = -945 / (32 * np.pi * h**9.0)  # constant component of gradient poly kernel
    dgradWPOLY_CONST = 315 / (64 * np.pi * h**9.0)  # constant component of gradient squared poly kernel
    gradWPRESS_CONST = -45.0 / (np.pi * h**6.0)     # constant component of gradient pressure kernel 
    dgradWVISCO_CONST = 45.0 / (np.pi * h**6.0)     # constant component of gradient squared viscocity kernel
    VISC = 0.001                                    # dynamic viscocity of water 0.001 [Pa*s] 
    RHO_0 = 1000                                    # rest density of water [kg/m3]
    SIGMA = 100                                     # viscocity multiplier
    Cp = 4.187 * 10**3                              # specific heat of water at constant pressure [J/kgK]
    GAMMA = 1.33                                    # specific heat ratio
    Cv = Cp / GAMMA                                 # specific heat of water at constant volume [J/kgK]
    TEMP = 288.15                                   # temperature at specfic heats [K]
    K_GAS = (GAMMA - 1) * Cv * TEMP                 # gas constant for equation of state
    m     = M/N                                     # single particle mass
    a     = 1500                                    # speed of sound in water [m/s]
    
    degp = bpy.context.evaluated_depsgraph_get()
    object = bpy.data.objects["Source"]
    particle_systems = object.evaluated_get(degp).particle_systems
    particles = particle_systems[0].particles
    totalParticles = len(particles)

    scene = bpy.context.scene
    cFrame = scene.frame_current
    sFrame = scene.frame_start
    
    #at start-frame, clear the particle cache
    if cFrame == sFrame:
        psSeed = object.particle_systems[0].seed 
        object.particle_systems[0].seed  = psSeed
    
        
    pos = [tuple(particle.location) for (index, particle) in particles.items()]  
    vel = [tuple(particle.velocity) for (index, particle) in particles.items()]
    
    pos = np.matrix(pos)
    vel = np.matrix(vel)
    
    #print(pos[0])
    # time constant (increase denom for slower sim rate)
    t_f = cFrame / 1.0
    # timestep
    dt = 0.02
    
    # get density
    rho = getDensity( pos, pos, m, h, WPOLY_CONST )
    # get forces
    force = getForce( pos, vel, m, h, G, 
                        IN, OUT, WPOLY_CONST, gradWPOLY_CONST, 
                        dgradWPOLY_CONST, gradWPRESS_CONST, 
                        dgradWVISCO_CONST, K_GAS, RHO_0, SIGMA, VISC, a )
    # (1/2) kick
    vel += dt/2 * force
    # update position
    pos += vel * dt
    #force = getForce( pos, vel, m, h, k, n )
    # (1/2) kick again
    vel += dt/2 * force

    idxi = 0
    for i in particles:
        # computing new particle position and velocity
        p = pos[idxi, :]
        v = vel[idxi, :] 
        # update data
        i.location = [pos[idxi, 0], pos[idxi, 1], pos[idxi, 2]]
        i.velocity = [vel[idxi, 0], vel[idxi, 1], vel[idxi, 2]]
        
        idxi += 1
        
    
def Wpoly( x, y, z, h, WPOLY_CONST ):
    """
    Gausssian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    w     is the evaluated smoothing function
    """
    r = np.sqrt(np.abs(x**2) + np.abs(y**2) + np.abs(z**2))
    #r2 = x**2 + y**2 + z**2

    w = WPOLY_CONST * ((h**2) -  r**2)**3 
    #w = WPOLY_CONST * ((h**2) -  r2)**3 
    
    return w

def gradWpoly( x, y, z, h, gradWPOLY_CONST ):
    """
    Gradient of the Gausssian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    wx, wy, wz     is the evaluated gradient
    """
    
    r = np.sqrt(np.abs(x**2) + np.abs(y**2) + np.abs(z**2))

    n = gradWPOLY_CONST * ((h**2) -  r**2)**2
   
    wx = n * x
    wy = n * y
    wz = n * z
    
    return wx, wy, wz
    
def dgradWpoly( x, y, z, h, dgradWPOLY_CONST ):

    r = np.sqrt(np.abs(x**2) + np.abs(y**2) + np.abs(z**2))
    #r2 = x**2 + y**2 + z**2

    w = dgradWPOLY_CONST * ((h**2) -  r**2) * (7 * r**2 - 3 * h**2)
    #w = dgradWPOLY_CONST * ((h**2) -  r2) * (7 * r2 - 3 * h**2)
    
    return w
    
def gradWpress( x, y, z, h, gradWPRESS_CONST ):
    
    r = np.sqrt(np.abs(x**2) + np.abs(y**2) + np.abs(z**2))
    #r = np.sqrt(x**2 + y**2 + z**2)
    #r = np.power((np.power(x, 2) + np.power(y, 2) + np.power(z, 2)), 0.5)
    
    #n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
    n = gradWPRESS_CONST * (h -  r)**2 / r
   
    wx = n * x
    wy = n * y
    wz = n * z
    
    return wx, wy, wz

def dgradWvisco( x, y, z, h, dgradWVISCO_CONST ):
    
    #r = np.sqrt(x**2 + y**2 + z**2)
    #r = np.power((np.power(x, 2) + np.power(y, 2) + np.power(z, 2)), 0.5)
    r = np.sqrt(np.abs(x**2) + np.abs(y**2) + np.abs(z**2))
    
    #n = Wvisco * (np.ones(len(r)) * h -  np.power(r, 2))
    n = dgradWVISCO_CONST * (h -  r)
    
    wx = n * x
    wy = n * y
    wz = n * z
    
    return wx, wy, wz
    
    
def getPairwiseSeparations( ri, rj ):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """
    
    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y,z)
    rix = ri[:,0].reshape((M,1))
    riy = ri[:,1].reshape((M,1))
    riz = ri[:,2].reshape((M,1))
    
    # other set of points positions rj = (x,y,z)
    rjx = rj[:,0].reshape((N,1))
    rjy = rj[:,1].reshape((N,1))
    rjz = rj[:,2].reshape((N,1))
    
    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
    
    return dx, dy, dz
    

def getDensity( r, pos, m, h, WPOLY_CONST ):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of accelerations
    dx, dy, dz are M x N
    """
    
    M = r.shape[0]
    
    dx, dy, dz = getPairwiseSeparations( r, pos );
    
    rho = np.sum( m * Wpoly(dx, dy, dz, h, WPOLY_CONST), 1 ).reshape((M,1))
    
    return rho
    
    
def getPressure(rho, K_GAS, RHO_0):
    """
    Equation of State
    rho     vector of densities
    K_GAS   equation of state constant
    RHO_0   rest density
    P       pressure
    """
    
    P = (rho - RHO_0) * K_GAS 
    
    return P


def getForce( pos, vel, m, h, G, IN, OUT, WPOLY_CONST, gradWPOLY_CONST, dgradWPOLY_CONST, 
                gradWPRESS_CONST, dgradWVISCO_CONST, K_GAS, RHO_0, SIGMA, VISC, a ):
    
    N = pos.shape[0]
    
    # Calculate densities at the position of the particles
    rho = getDensity( pos, pos, m, h, WPOLY_CONST )
    
    # Get the pressures
    P = getPressure(rho, K_GAS, RHO_0)
    
    # Get pairwise distances and gradients
    "ri - rj"
    dx, dy, dz = getPairwiseSeparations( pos, pos )
    "vj - vi"
    dvx, dvy, dvz = getPairwiseSeparations( -vel, -vel )
    "gradient terms for poly, pressure, and viscocity"
    ddWpoly = dgradWpoly( dx, dy, dz, h, dgradWPOLY_CONST )
    dWx, dWy, dWz = gradWpoly( dx, dy, dz, h, gradWPOLY_CONST )
    dWxp, dWyp, dWzp = gradWpress( dx, dy, dz, h, gradWPRESS_CONST )
    ddWxv, ddWyv, ddWzv = dgradWvisco( dx, dy, dz, h, dgradWVISCO_CONST )
    
    "Pressure forces"
    fxp = - np.sum( m / (rho * rho.T) * ( P/2 + P.T/2 ) * dWxp + IN * dx, 1).reshape((N,1))
    fyp = - np.sum( m / (rho * rho.T) * ( P/2 + P.T/2 ) * dWyp + IN * dy, 1).reshape((N,1))
    fzp = - np.sum( m / (rho * rho.T) * ( P/2 + P.T/2 ) * dWzp + IN * dz, 1).reshape((N,1)) 
    
    "Viscocity forces"
    fxv = np.sum( m / (rho * rho.T) * VISC * dvx * ddWxv - (IN * OUT) * dvx, 1).reshape((N,1))
    fyv = np.sum( m / (rho * rho.T) * VISC * dvy * ddWyv - (IN * OUT) * dvy, 1).reshape((N,1))
    fzv = np.sum( m / (rho * rho.T) * VISC * dvz * ddWzv - (IN * OUT) * dvz, 1).reshape((N,1))
    
    "Surface/interface tension constants and forces"
    dgradC = m * a / (rho * rho.T) * ddWpoly
    gradCx = np.sum( m * a / (rho * rho.T) * dWx, 1).reshape((N,1)) 
    gradCy = np.sum( m * a / (rho * rho.T) * dWy, 1).reshape((N,1)) 
    gradCz = np.sum( m * a / (rho * rho.T) * dWz, 1).reshape((N,1)) 
    n = np.hstack((gradCx, gradCy, gradCz))
    n_norm = np.linalg.norm(n)
    # use minus to get positive curvature
    curv_k = - dgradC / n_norm
    force_surf = -SIGMA * curv_k * n
        
    "Total forces"
    fx = fxp + fxv 
    fy = fyp + fyv 
    fz = fzp + fzv + rho*G[2] 
    scale = 10*rho
    # pack together the acceleration components
    force = np.hstack((fx, fy, fz/rho)) + force_surf 
    
    return force
        
        
"Registration and Unregistration"
classes = [Properties, SPH_PT_Panel, SPH_OT_Operator]

def register():
    for cls in classes:
        
        bpy.utils.register_class(cls)
        bpy.types.Scene.sph_tool = bpy.props.PointerProperty(type = Properties)


def unregister():
    for cls in classes:
        
        bpy.utils.unregister_class(cls)
        bpy.types.Scene.my_tool = bpy.props.PointerProperty(type = Properties)
        del bpy.types.Scene.sph_tool


if __name__ == "__main__":
    register()