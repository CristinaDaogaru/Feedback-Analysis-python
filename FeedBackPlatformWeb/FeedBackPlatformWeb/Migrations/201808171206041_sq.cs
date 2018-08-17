namespace FeedBackPlatformWeb.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class sq : DbMigration
    {
        public override void Up()
        {
            CreateTable(
                "dbo.Categories",
                c => new
                    {
                        Id = c.Int(nullable: false, identity: true),
                        Name = c.String(maxLength: 20),
                    })
                .PrimaryKey(t => t.Id);
            
            CreateTable(
                "dbo.Surveys",
                c => new
                    {
                        Id = c.Int(nullable: false, identity: true),
                        Name = c.String(maxLength: 20),
                        CategoryId = c.Int(nullable: false),
                    })
                .PrimaryKey(t => t.Id)
                .ForeignKey("dbo.Categories", t => t.CategoryId, cascadeDelete: true)
                .Index(t => t.CategoryId);
            
            CreateTable(
                "dbo.ClientProfiles",
                c => new
                    {
                        IdClient = c.Int(nullable: false, identity: true),
                        Name = c.String(maxLength: 20),
                        Email = c.String(),
                        Password = c.String(nullable: false, maxLength: 100),
                    })
                .PrimaryKey(t => t.IdClient);
            
            CreateTable(
                "dbo.Questions",
                c => new
                    {
                        Id = c.Int(nullable: false, identity: true),
                        Description = c.String(),
                    })
                .PrimaryKey(t => t.Id);
            
            CreateTable(
                "dbo.Responses",
                c => new
                    {
                        Id = c.Int(nullable: false, identity: true),
                        QuestionId = c.Int(nullable: false),
                        Content = c.String(),
                        IsChecked = c.Boolean(nullable: false),
                    })
                .PrimaryKey(t => t.Id)
                .ForeignKey("dbo.Questions", t => t.QuestionId, cascadeDelete: true)
                .Index(t => t.QuestionId);
            
        }
        
        public override void Down()
        {
            DropForeignKey("dbo.Responses", "QuestionId", "dbo.Questions");
            DropForeignKey("dbo.Surveys", "CategoryId", "dbo.Categories");
            DropIndex("dbo.Responses", new[] { "QuestionId" });
            DropIndex("dbo.Surveys", new[] { "CategoryId" });
            DropTable("dbo.Responses");
            DropTable("dbo.Questions");
            DropTable("dbo.ClientProfiles");
            DropTable("dbo.Surveys");
            DropTable("dbo.Categories");
        }
    }
}
