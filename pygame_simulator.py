import pygame as pg

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)


class PyGame:
    def __init__(self, n_agents, map_info):
        pg.init()
        # start up dat pygame
        pg.display.set_caption("Multi Robot Environment in Factory")
        screen_size = [1400, 850]
        self.screen = pg.display.set_mode(screen_size)
        self.side_length = 20
        self.margin_length = 2
        # the map should locate in the center of the screen
        self.map_init_x = screen_size[0]/2 - \
                          (self.side_length+self.margin_length) * n_agents/2
        self.map_init_y = (screen_size[1] - (
                self.side_length + self.margin_length) * n_agents) / 2
        self.texts = []
        self.text_position = []
        self.n_agents = n_agents
        self.map = map_info
        self.clock = pg.time.Clock()
        self.rotation_angle = 0
        self.event = pg.event.get()
        self.QUIT = pg.QUIT
        self.display = pg.display

    def draw_map(self, row_high, column_high):
        for x in range(row_high + 1):
            for y in range(column_high + 1):
                color = WHITE
                if self.map[x][y] == -1:
                    color = BLACK
                elif self.map[x][y] == 1:
                    color = BLUE
                elif self.map[x][y] == 2:
                    color = GREEN
                apex_x = self.map_init_x + y * (self.side_length + self.margin_length)
                apex_y = self.map_init_y + x * (self.side_length + self.margin_length)
                pg.draw.rect(self.screen, color,
                             [apex_x, apex_y, self.side_length, self.side_length])

    def draw_rect(self, x, y, color=GREEN, draw_dynamic=False, le=0):
        apex_x = self.map_init_x + y * (self.side_length + self.margin_length)
        apex_y = self.map_init_y + x * (self.side_length + self.margin_length)
        if not draw_dynamic:
            pg.draw.rect(self.screen, color,
                         [apex_x, apex_y, self.side_length, self.side_length])
        else:
            pg.draw.rect(self.screen, color,
                             [apex_x, apex_y, le*self.side_length, self.side_length])

    # show whether agent carry stock, whether front worker produce stock,
    # whether back worker accept stock
    def draw_circle(self, x, y, color=RED):
        pg.draw.circle(
            self.screen, color,
            (self.map_init_x + y * (self.side_length + self.margin_length) +
             self.side_length / 2, self.map_init_y +
             x * (self.side_length + self.margin_length) + self.side_length / 2), 10)

    def rend_text(self, to_display, x, y, color=BLUE,
                  color_background=WHITE, font_size=30):
        basic_font = pg.font.SysFont(None, font_size)
        text = basic_font.render(to_display, True, color, color_background)
        pos = text.get_rect(
            x=self.map_init_x + y * (self.side_length + self.margin_length),
            y=self.map_init_y + x * (self.side_length + self.margin_length))
        self.screen.blit(text, pos)

    def rend_image(self, image_path, x, y, scale=1, is_working=False):
        image = pg.image.load(image_path)
        size = image.get_size()
        size = (size[0] * scale, size[1] * scale)
        image = pg.transform.scale(image, size)
        if is_working:
            self.rotation_angle += 1
            if self.rotation_angle >= 360:
                self.rotation_angle = 0
            image = pg.transform.rotate(image, self.rotation_angle)
        pos = image.get_rect(
            x=self.map_init_x + y * (self.side_length + self.margin_length),
            y=self.map_init_y + x * (self.side_length + self.margin_length))
        self.screen.blit(image, pos)

