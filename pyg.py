import pygame

import get_img

pygame.init()
 
# Set the width and height of the screen [width,height]
size = [100, 100]

screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

done = False

pygame.joystick.init()


# -------- Main Program Loop -----------
while done==False:
    # EVENT PROCESSING STEP
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done=True # Flag that we are done so we exit this loop
        
        # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")
            

    # Get count of joysticks
    joystick_count = pygame.joystick.get_count()
    
    # For each joystick:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
    
        name = joystick.get_name()

        axes = joystick.get_numaxes()
        
        axis = joystick.get_axis(0)

        bbox = (1700,150,2560,600)

        get_img.take_screenshot(name=axis,bbox=bbox)

        
pygame.quit()
